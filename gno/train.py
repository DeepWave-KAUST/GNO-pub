# -----------------------------------------------------------------------------
# Title: Training GNO
# Description:
#     This script trains a GNO (Guided Noise Operator) diffusion model to
#     represent the frequency-domain scattered wavefield conditioned on a
#     background wavefield (u0) and a velocity model (vel).
#     The training process involves:
#       - Loading pre-generated synthetic training data from a directory.
#       - Building the GNO model and its associated Gaussian diffusion process.
#       - Selecting a timestep schedule sampler (uniform or loss-aware).
#       - Running the training loop with optional EMA, FP16, and checkpointing.
# Author: Shijun Cheng
# Acknowledgment:
#     This code is adapted from the Improved DDPM (IDDPM) framework.
#     IDDPM GitHub: https://github.com/openai/improved-diffusion
#     IDDPM Paper: https://arxiv.org/abs/2102.09672
# -----------------------------------------------------------------------------

import argparse
from code import logger
from code.datasets import load_data
from code.resample import create_named_schedule_sampler
from code.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.train_util import TrainLoop
import torch as th


def main():
    """
    Entry point for GNO diffusion model training.

    Workflow:
        1. Parse command-line arguments (hyperparameters and model settings).
        2. Initialize the GNO model and Gaussian diffusion process on GPU.
        3. Build a timestep schedule sampler that determines which noise levels
           are prioritized during each training iteration.
        4. Create a streaming data loader over the synthetic training dataset.
        5. Launch the TrainLoop, which handles gradient updates, EMA weight
           tracking, optional FP16 mixed-precision training, periodic logging,
           and checkpoint saving/resuming.

    Note:
        During training, ``timestep_respacing`` should be set to "" (empty) in
        ``model_and_diffusion_defaults`` so that the full T-step diffusion chain
        is used. DDIM step schedules (e.g., "DDIM50") are only applied at
        inference time in sample.py.
    """
    # Parse all hyperparameters and model configuration from the command line
    args = create_argparser().parse_args()

    # Initialize logger (writes to stdout and optionally to a log file)
    logger.configure()

    # All training is performed on GPU
    device = th.device('cuda')

    # -------------------------------------------------------------------------
    # Model and diffusion setup
    # -------------------------------------------------------------------------
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)

    # -------------------------------------------------------------------------
    # Timestep schedule sampler
    # -------------------------------------------------------------------------
    # The schedule sampler controls which diffusion timestep t is drawn for
    # each training sample:
    #   "uniform"      : sample t ~ Uniform{1, ..., T}; simple and stable.
    #   "loss-second-moment" : up-weight timesteps with high recent loss
    #                          variance, focusing capacity on hard noise levels.
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # -------------------------------------------------------------------------
    # Data loader
    # -------------------------------------------------------------------------
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,      # Root directory of synthetic training .mat files
        batch_size=args.batch_size,  # Number of (u0, du, vel) triplets per batch
        class_cond=args.class_cond,  # Whether to pass class labels as extra conditioning
                                     # (set False for unconditional/wavefield-conditioned training)
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    logger.log("training...")
    TrainLoop(
        model=model,                            # GNO denoising network
        diffusion=diffusion,                    # Gaussian diffusion (forward + reverse)
        data=data,                              # Streaming data loader
        batch_size=args.batch_size,             # Batch size (must match data loader)
        lr=args.lr,                             # Initial learning rate for AdamW optimizer
        ema_rate=args.ema_rate,                 # EMA decay rate(s) for model weights;
                                                # EMA weights are used at inference to
                                                # produce smoother, more stable predictions
        log_interval=args.log_interval,         # Print training metrics every N steps
        save_interval=args.save_interval,       # Save model checkpoint every N steps
        resume_checkpoint=args.resume_checkpoint,  # Path to .pt checkpoint to resume
                                                   # training; "" means train from scratch
        use_fp16=args.use_fp16,                 # Enable FP16 mixed-precision to reduce
                                                # memory and accelerate training on A100/V100
        fp16_scale_growth=args.fp16_scale_growth,  # Rate at which the FP16 loss scale grows
                                                   # when no overflow is detected
        schedule_sampler=schedule_sampler,      # Timestep sampling strategy (see above)
        weight_decay=args.weight_decay,         # L2 weight decay for AdamW regularization
        lr_anneal_steps=args.lr_anneal_steps,   # Total steps over which lr decays to 0;
                                                # training typically continues beyond this
                                                # point with a near-zero learning rate
    ).run_loop()


def create_argparser():
    """
    Build the command-line argument parser with default hyperparameters.

    Default values are merged with architecture/diffusion settings from
    ``model_and_diffusion_defaults()``, so all model hyperparameters (e.g.,
    num_res_blocks, attention_resolutions, timestep_respacing) can also be
    overridden from the command line.

    Key arguments:
        data_dir          (str)   : Path to the directory of synthetic training
                                    data (.mat files with u0, du, vel fields).
        schedule_sampler  (str)   : Timestep sampling strategy; "uniform" is
                                    recommended for stable early training.
        lr                (float) : Initial learning rate; 1e-4 works well for
                                    Adam-family optimizers on this architecture.
        weight_decay      (float) : L2 regularization coefficient for AdamW.
        lr_anneal_steps   (int)   : Linear decay schedule length; lr reaches 0
                                    at this step count.
        batch_size        (int)   : Training mini-batch size.
        ema_rate          (str)   : Comma-separated EMA decay values, e.g.
                                    "0.999" or "0.999,0.9999". Multiple values
                                    produce multiple EMA checkpoints in parallel.
        log_interval      (int)   : Frequency of console/log output (in steps).
        save_interval     (int)   : Frequency of checkpoint saving (in steps).
        resume_checkpoint (str)   : Path to an existing .pt file to resume from;
                                    leave empty ("") to train from scratch.
        use_fp16          (bool)  : Enable mixed-precision (FP16) training.
        fp16_scale_growth (float) : Loss scale growth factor for FP16 stability.

    Returns:
        argparse.ArgumentParser: Fully configured argument parser.
    """
    defaults = dict(
        data_dir="../../../dataset/train_multifreq_multismooth/",   # Synthetic training data directory
        schedule_sampler="uniform",     # Timestep sampling: "uniform" or "loss-second-moment"
        lr=1e-4,                        # AdamW learning rate
        weight_decay=5e-5,              # AdamW weight decay (L2 regularization)
        lr_anneal_steps=600000,         # Steps for linear learning rate decay to zero
        batch_size=64,                  # Training batch size
        ema_rate="0.999",               # EMA decay rate(s); higher = smoother but slower tracking
        log_interval=100,               # Log training loss every 100 steps
        save_interval=10000,            # Save checkpoint every 10,000 steps
        resume_checkpoint="",           # Resume path; empty string = train from scratch
        use_fp16=False,                 # Mixed-precision training (recommended for A100)
        fp16_scale_growth=1e-3,         # FP16 loss scale growth rate
    )
    # Merge with architecture and diffusion hyperparameters (e.g., model depth,
    # number of diffusion steps, noise schedule type)
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # Dynamically register all keys in defaults as CLI flags
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()