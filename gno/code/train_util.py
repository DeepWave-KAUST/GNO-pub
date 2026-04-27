import copy
import functools
import os
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from . import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import random
import time


# Initial log2 loss scale for FP16 dynamic loss scaling.
# A value of 20 corresponds to a loss scale of 2^20 ≈ 1e6, which is large
# enough to prevent FP16 underflow at the start of training while still
# leaving headroom for the scale to decrease if NaNs are detected.
INITIAL_LOG_LOSS_SCALE = 20.0

# Directory where model and EMA checkpoints are saved
dir_checkpoints = './checkpoints/'
os.makedirs(dir_checkpoints, exist_ok=True)


class TrainLoop:
    """
    Main training loop for the GNO diffusion model.

    Handles the complete training lifecycle including:
        - Checkpoint loading and resuming.
        - FP16 mixed-precision setup (optional).
        - AdamW optimization with linear learning rate annealing.
        - Exponential Moving Average (EMA) weight tracking.
        - Periodic logging and checkpoint saving.

    The loop iterates over batches of (du, u0, vel) triplets drawn from the
    infinite data generator produced by load_data(). At each step it:
        1. Samples a random diffusion timestep t for each item in the batch.
        2. Computes the diffusion training loss (MSE on predicted x_0 or epsilon).
        3. Back-propagates and updates parameters (with FP16 scaling if enabled).
        4. Updates all EMA parameter copies.
        5. Logs metrics and saves checkpoints at the configured intervals.

    Args:
        model             (nn.Module)        : GNO denoising UNet.
        diffusion         (GaussianDiffusion): Diffusion process (forward + loss).
        data              (generator)        : Infinite (du, u0, vel, cond) generator.
        batch_size        (int)              : Training mini-batch size.
        lr                (float)            : Initial AdamW learning rate.
        ema_rate          (float or str)     : EMA decay rate(s). A single float or
                                               a comma-separated string for multiple
                                               parallel EMA copies (e.g. "0.999,0.9999").
        log_interval      (int)              : Log metrics every N steps.
        save_interval     (int)              : Save checkpoint every N steps.
        resume_checkpoint (str)              : Path to a .pt checkpoint to resume from;
                                               "" means train from scratch.
        use_fp16          (bool)             : Enable FP16 mixed-precision training.
        fp16_scale_growth (float)            : Rate at which the FP16 loss scale grows
                                               per step when no overflow is detected.
        schedule_sampler  (ScheduleSampler)  : Timestep sampling strategy. Defaults to
                                               UniformSampler if None.
        weight_decay      (float)            : AdamW L2 weight decay coefficient.
        lr_anneal_steps   (int)              : Total steps for linear LR decay to zero.
                                               0 means no annealing (constant LR).
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=1e-4,
        lr_anneal_steps=0,
    ):
        self.model = model
        # Infer device from the first model parameter (avoids hard-coding "cuda")
        self.device = next(model.parameters()).device
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr

        # Normalize ema_rate to a list of floats so the rest of the code can
        # always iterate over it uniformly, regardless of how it was specified
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.log_interval = log_interval        # Print metrics every N steps
        self.save_interval = save_interval      # Write checkpoint every N steps
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        # Fall back to uniform timestep sampling if no sampler is provided
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        # Global step counter (incremented each iteration) and the step at
        # which training was resumed (read from checkpoint filename)
        self.step = 0
        self.resume_step = 0

        # Effective batch size (single-GPU; no DDP in this implementation)
        self.global_batch = self.batch_size

        # In FP32 mode master_params == model_params (same list).
        # In FP16 mode _setup_fp16() replaces master_params with a FP32 copy.
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        # Log2 of the current FP16 dynamic loss scale
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        # Flag used to decide whether to synchronize CUDA after certain ops
        self.sync_cuda = th.cuda.is_available()

        # Load checkpoint weights (if resuming) and extract resume_step
        self._load_and_sync_parameters()

        # If using FP16, create FP32 master params and convert model to FP16
        if self.use_fp16:
            self._setup_fp16()

        # AdamW optimizer operates on master_params (FP32 in both FP16 and FP32 modes)
        self.opt = th.optim.AdamW(
            self.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        if self.resume_step:
            # Restore optimizer state and EMA weights from matching checkpoints
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # Initialize EMA parameter copies from the current (random) model weights
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

    # -------------------------------------------------------------------------
    # Checkpoint loading helpers
    # -------------------------------------------------------------------------

    def _load_and_sync_parameters(self):
        """
        Load model weights from a checkpoint if one is available.

        Priority: auto-discovered checkpoint (find_resume_checkpoint()) >
                  manually specified self.resume_checkpoint.
        Also parses resume_step from the checkpoint filename so the global
        step counter continues from where training left off.
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(resume_checkpoint, map_location=self.device)
            )

    def _load_ema_parameters(self, rate):
        """
        Load EMA weights for a given decay rate from the matching checkpoint.

        If the expected EMA checkpoint file does not exist, returns a deep copy
        of the current master params (effectively reinitializing EMA from the
        resumed model weights).

        Args:
            rate (float): EMA decay rate (e.g. 0.999).

        Returns:
            list of Tensor: EMA parameter list in master-param format.
        """
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load_state_dict(ema_checkpoint, map_location=self.device)
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        """
        Restore AdamW optimizer state (momentum buffers, step counts) from
        the checkpoint that corresponds to resume_step.

        The optimizer checkpoint is expected at the same directory as the
        model checkpoint with filename opt{resume_step:06d}.pt.
        If the file does not exist, the optimizer starts from a fresh state.
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}"
            )
            state_dict = th.load_state_dict(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        Prepare for FP16 mixed-precision training.

        Creates a flat FP32 copy of all model parameters (master_params) that
        the optimizer will update, then converts the model weights in-place to
        FP16 for forward/backward passes. Gradients are accumulated in FP32
        master_params to preserve numerical precision.
        """
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def run_loop(self):
        """
        Main training loop.

        Iterates until self.step + self.resume_step reaches lr_anneal_steps
        (or indefinitely if lr_anneal_steps == 0). At each iteration:
            - Fetches the next batch from the data generator.
            - Runs one forward-backward-optimize step.
            - Logs metrics at log_interval.
            - Saves checkpoints at save_interval.

        A final checkpoint is saved after the loop exits if the last step was
        not already a save step.
        """
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # Fetch next batch: scattered wavefield, background wavefield,
            # velocity model, and optional conditioning dict
            batch_du, batch_u0, batch_vel, cond = next(self.data)

            self.run_step(batch_du, batch_u0, batch_vel, cond)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                # Early exit hook for integration / smoke tests
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1

        # Ensure the final model state is always saved even if the loop exits
        # between two regular save_interval checkpoints
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_du, batch_u0, batch_vel, cond):
        """
        Execute a single training step: forward pass, backward pass, and
        parameter update.

        Args:
            batch_du  (Tensor): Scattered wavefield batch, shape (B, 2, nz, nx).
            batch_u0  (Tensor): Background wavefield batch, shape (B, 2, nz, nx).
            batch_vel (Tensor): Velocity model batch, shape (B, 1, nz, nx).
            cond      (dict)  : Optional conditioning dict (e.g. class labels).
        """
        self.forward_backward(batch_du, batch_u0, batch_vel, cond)
        if self.use_fp16:
            self.optimize_fp16()   # FP16 path: gradient unscaling + overflow check
        else:
            self.optimize_normal() # FP32 path: standard gradient step
        self.log_step()

    def forward_backward(self, batch_du, batch_u0, batch_vel, cond):
        """
        Compute the diffusion training loss and accumulate gradients.

        Workflow:
            1. Zero all parameter gradients.
            2. Move batch tensors to the training device.
            3. Sample a random diffusion timestep t for each sample in the batch
               using the schedule sampler.
            4. Compute training_losses() — MSE on x_0 or epsilon predictions,
               weighted by the schedule sampler importance weights.
            5. If using a LossAwareSampler, update its per-timestep loss
               statistics with the current batch losses.
            6. Back-propagate; scale the loss by 2^lg_loss_scale in FP16 mode
               to prevent gradient underflow.

        Args:
            batch_du  (Tensor): Scattered wavefield (training target).
            batch_u0  (Tensor): Background wavefield (conditioning).
            batch_vel (Tensor): Velocity model (conditioning).
            cond      (dict)  : Additional model kwargs (e.g. class labels).
        """
        zero_grad(self.model_params)

        # Move all inputs to the training device
        batch_du  = batch_du.to(self.device)
        batch_u0  = batch_u0.to(self.device)
        batch_vel = batch_vel.to(self.device)

        # Sample diffusion timesteps; t shape: (B,), weights shape: (B,)
        # Weights are 1/p(t) for importance-weighted loss averaging
        t, weights = self.schedule_sampler.sample(batch_du.shape[0], self.device)

        # Partially apply diffusion.training_losses so it can be called with no
        # arguments (supports gradient checkpointing in the UNet if enabled)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch_du,
            batch_u0,
            t,
            batch_vel,
            model_kwargs=cond,
        )

        losses = compute_losses()

        # Update per-timestep loss statistics for loss-aware timestep sampling
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        # Importance-weighted mean loss across the batch
        loss = (losses["loss"] * weights).mean()

        # Log per-key and per-quartile loss statistics
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )

        # Backward pass; scale loss in FP16 mode to avoid gradient underflow
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    # -------------------------------------------------------------------------
    # Optimization steps
    # -------------------------------------------------------------------------

    def optimize_fp16(self):
        """
        Perform an optimizer step in FP16 mixed-precision mode.

        Workflow:
            1. Check all FP16 model gradients for NaN/Inf (overflow).
               If overflow is detected, reduce the loss scale and skip the step.
            2. Copy FP16 gradients to the FP32 master params.
            3. Un-scale gradients by dividing by the current loss scale.
            4. Log the global gradient norm.
            5. Apply learning rate annealing.
            6. Run the AdamW optimizer step on FP32 master params.
            7. Update all EMA copies from the updated master params.
            8. Copy updated FP32 master params back to the FP16 model weights.
            9. Grow the loss scale for the next step (if no overflow occurred).
        """
        # Overflow detection: skip step and reduce scale if any gradient is non-finite
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(
                f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}"
            )
            return

        # Copy FP16 gradients → FP32 master params for numerically stable update
        model_grads_to_master_grads(self.model_params, self.master_params)

        # Undo the loss scaling applied in forward_backward()
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))

        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()

        # Update every EMA copy with the freshly updated master params
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

        # Sync updated FP32 master params back into the FP16 model
        master_params_to_model_params(self.model_params, self.master_params)

        # Grow loss scale for next step (no overflow this iteration)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """
        Perform a standard FP32 optimizer step.

        Logs gradient norm, applies LR annealing, runs AdamW, and updates
        all EMA parameter copies.
        """
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _log_grad_norm(self):
        """
        Compute and log the global L2 gradient norm across all master params.

        The norm is logged as a running mean under the key "grad_norm".
        Monitoring this value helps detect gradient explosion or vanishing
        during training.
        """
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is None:
                continue
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """
        Apply linear learning rate annealing based on training progress.

        The effective LR follows:
            lr_effective = lr * (1 - 0.8 * step / lr_anneal_steps)

        The 0.8 factor means the LR reaches 20% of its initial value at
        lr_anneal_steps rather than zero, providing a non-zero floor.
        No-op if lr_anneal_steps == 0 (constant LR).
        """
        if not self.lr_anneal_steps:
            return
        frac_done = 0.8 * (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """
        Log per-step training statistics to the logger.

        Always logs:
            step    : Global step count (resume_step + current step).
            samples : Total number of training samples seen so far.
        In FP16 mode also logs:
            lg_loss_scale : Current log2 of the dynamic loss scale.
        """
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv(
            "samples",
            (self.step + self.resume_step + 1) * self.global_batch,
        )
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """
        Save EMA model checkpoints to dir_checkpoints.

        One checkpoint file is written per EMA rate:
            ema_{rate}_{step:06d}.pt

        Note: Saving the raw model weights (master_params) and the optimizer
        state is currently commented out to reduce storage overhead on long
        training runs. Uncomment the corresponding blocks if full resumability
        is required.
        """
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                # Raw (non-EMA) model checkpoint
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                # EMA checkpoint named by decay rate and global step
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(dir_checkpoints, filename), "wb") as f:
                th.save(state_dict, f)

        # Save only EMA checkpoints (raw model save is intentionally disabled)
        # Uncomment save_checkpoint(0, self.master_params) to also save raw weights
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Optimizer state saving is disabled to save disk space.
        # Uncomment the block below to re-enable full training resumability:
        # with bf.BlobFile(
        #     bf.join(dir_checkpoints, f"opt{(self.step+self.resume_step):06d}.pt"),
        #     "wb",
        # ) as f:
        #     th.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        """
        Convert master params (possibly a flat FP32 list) back to a named
        state dict compatible with model.load_state_dict().

        In FP16 mode, master_params is a flattened list; unflatten_master_params
        reshapes it to match the model's parameter tensors before copying.

        Args:
            master_params (list of Tensor): FP32 master parameter list.

        Returns:
            OrderedDict: Named state dict ready for serialization with th.save().
        """
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        Convert a named state dict back to a master-param list.

        In FP16 mode, wraps the tensors in a flat FP32 master-param structure
        via make_master_params(). In FP32 mode, returns a plain list of tensors.

        Args:
            state_dict (OrderedDict): Named state dict loaded from a checkpoint.

        Returns:
            list of Tensor: Master parameter list in the format expected by
                            the optimizer and EMA update functions.
        """
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


# -----------------------------------------------------------------------------
# Module-level utility functions
# -----------------------------------------------------------------------------

def parse_resume_step_from_filename(filename):
    """
    Extract the training step count from a checkpoint filename.

    Expected filename format: path/to/modelNNNNNN.pt, where NNNNNN is a
    zero-padded integer representing the number of completed training steps.

    Args:
        filename (str): Path to a model checkpoint file.

    Returns:
        int: Parsed step count, or 0 if parsing fails.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def parse_dataname_from_filename(filename):
    """
    Extract a dataset name suffix from a checkpoint filename.

    Expected filename format: path/to/gaussian5<suffix>.pt, where <suffix>
    identifies the dataset variant used during training.

    Args:
        filename (str): Path to a checkpoint file.

    Returns:
        str: Parsed dataset name suffix, or 0 if parsing fails.
    """
    split = filename.split("gaussian5")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return split1
    except ValueError:
        return 0


def get_blob_logdir():
    """
    Return the directory used for blob (cloud) logging.

    Reads the DIFFUSION_BLOB_LOGDIR environment variable; falls back to the
    local logger directory if the variable is not set.

    Returns:
        str: Path to the blob log directory.
    """
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    """
    Auto-discover the latest checkpoint for resuming training.

    Currently returns None (no auto-discovery). Override this function to
    implement automatic checkpoint detection from blob storage or a shared
    file system (e.g., scanning dir_checkpoints for the highest-step .pt file).

    Returns:
        str or None: Path to the latest checkpoint, or None if not found.
    """
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    Locate the EMA checkpoint that corresponds to a given model checkpoint.

    Constructs the expected EMA filename ema_{rate}_{step:06d}.pt in the same
    directory as main_checkpoint and checks whether it exists.

    Args:
        main_checkpoint (str or None): Path to the main model checkpoint.
        step            (int)        : Training step of the checkpoint.
        rate            (float)      : EMA decay rate (e.g. 0.999).

    Returns:
        str or None: Path to the EMA checkpoint if it exists, else None.
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Log per-key loss values and their per-quartile breakdowns.

    For each loss key (e.g. "loss", "mse", "vb"):
        - Logs the batch-mean value under key.
        - Buckets each sample's timestep into one of four quartiles
          [0, T/4), [T/4, T/2), [T/2, 3T/4), [3T/4, T) and logs the
          per-quartile mean under "{key}_q{0-3}".

    Quartile logging helps identify which noise levels (early vs. late
    diffusion steps) contribute most to the overall training loss,
    which is useful for diagnosing schedule or architecture issues.

    Args:
        diffusion: GaussianDiffusion instance (provides num_timesteps).
        ts      (Tensor)           : Sampled timesteps, shape (B,).
        losses  (dict of Tensors)  : Loss tensors, each of shape (B,),
                                     already multiplied by importance weights.
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log per-quartile loss for diagnosing noise-level-specific behaviour
        for sub_t, sub_loss in zip(
            ts.cpu().numpy(), values.detach().cpu().numpy()
        ):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)