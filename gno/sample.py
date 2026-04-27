# -----------------------------------------------------------------------------
# Title: Seismic scattered wavefield representation with GNO
# Description:
#     This script performs inference/sampling on test data using a trained GNO
#     (Guided Noise Operator) diffusion model. The model takes background
#     wavefields (u0) and velocity models (vel) as conditions and generates
#     the corresponding scattered wavefields (du).
#     Supports both DDPM (full diffusion) and DDIM (accelerated deterministic)
#     sampling, with an optional PDE-guided posterior (Helmholtz constraint).
# Author: Shijun Cheng
# -----------------------------------------------------------------------------

import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from code.datasets import normalizer_vel
import scipy.io as sio
from code import logger
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import random
import math


def main(freq):
    """
    Main inference function for scattered wavefield generation at a given frequency.

    Workflow:
        1. Parse command-line arguments.
        2. Load the trained GNO diffusion model onto GPU.
        3. Iterate over multiple test datasets (different velocity model types).
        4. For each dataset, load background wavefields and velocity models,
           then run DDPM or DDIM sampling to predict the scattered wavefield.
        5. Optionally apply PDE guidance (Helmholtz equation constraint) during
           sampling to enforce physical consistency.
        6. Evaluate prediction quality via MSE against the ground-truth scattered
           wavefield and save all outputs to .mat files.

    Args:
        freq (int or float): Source frequency (Hz) used for both the angular
            frequency omega = 2*pi*freq and for locating the corresponding
            test data files.

    Notes on sampling modes:
        - DDPM  : set ``use_ddim=False``; uses the full reverse diffusion chain
                  (T steps). Recommended during training (timestep_respacing="").
        - DDIM  : set ``use_ddim=True`` and ``timestep_respacing="DDIMN"``,
                  where N controls the number of sampling steps (N-1 denoising
                  iterations). Provides a large speed-up over DDPM at inference.
        - PDE guidance : set ``pde_guide=True`` in ``code/script_util.py``.
                  ``scale_factor`` in create_argparser controls the guidance
                  strength — larger values enforce the Helmholtz PDE more
                  aggressively but may reduce sample diversity.
    """
    # -------------------------------------------------------------------------
    # 1. Argument parsing and device setup
    # -------------------------------------------------------------------------
    args = create_argparser().parse_args()

    # All computation is performed on GPU
    device = th.device('cuda')

    # Initialize logger for console output and result checkpointing
    logger.configure()

    # -------------------------------------------------------------------------
    # 2. Model instantiation and checkpoint loading
    # -------------------------------------------------------------------------
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # Load pre-trained weights; map_location ensures compatibility across GPUs
    model.load_state_dict(
        th.load(f'{args.model_path}', map_location=device)
    )
    model.to(device=device)
    model.eval()  # Disable BatchNorm / Dropout for deterministic inference

    # MSE criterion for evaluating real and imaginary parts independently
    criterion = th.nn.MSELoss()
    logger.log("sampling...")

    # -------------------------------------------------------------------------
    # 3. Frequency-domain parameters
    # -------------------------------------------------------------------------
    # Angular frequency omega = 2*pi*f, used in the Helmholtz PDE constraint:
    #   (∇² + omega²/v²) u = -s
    omega = 2 * math.pi * freq

    # -------------------------------------------------------------------------
    # 4. Output directory setup
    # -------------------------------------------------------------------------
    # Directory structure encodes sampling method and PDE guidance configuration,
    # making it straightforward to compare different experimental settings.
    if not args.use_ddim:
        # DDPM branch: full reverse diffusion chain
        if args.pde_guide:
            # PDE-guided DDPM; scale_factor controls guidance strength
            dir_output = f'./output/ddpm/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            # Standard DDPM without physics guidance
            dir_output = f'./output/ddpm/usepde_{args.pde_guide}/'
    else:
        # DDIM branch: accelerated sampling with N-1 steps (timestep_respacing="DDIMN")
        if args.pde_guide:
            # PDE-guided DDIM; scale_factor controls guidance strength
            dir_output = f'./output/{args.timestep_respacing}/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            # Standard DDIM without physics guidance
            dir_output = f'./output/{args.timestep_respacing}/usepde_{args.pde_guide}/'
    os.makedirs(dir_output, exist_ok=True)

    # Placeholder for optional classifier or external conditioning signals
    # (unused here, but required by the sampling API)
    model_kwargs = {}

    # -------------------------------------------------------------------------
    # 5. Select sampling function
    # -------------------------------------------------------------------------
    # p_sample_loop  : standard DDPM reverse diffusion (slow, stochastic)
    # ddim_sample_loop: DDIM deterministic sampler (fast, fewer NFEs)
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    # -------------------------------------------------------------------------
    # 6. Iterate over test datasets
    # -------------------------------------------------------------------------
    # Four benchmark velocity model families covering diverse geological structures
    md_list = ['CurveVel-A', 'CurveFault-A', 'CurveFault-B', 'FlatFault-B']

    for md in md_list:
        print(f'Sampling for test data {md} with frequency {freq}')

        # ---------------------------------------------------------------------
        # 6a. Load test data from .mat file
        # ---------------------------------------------------------------------
        # Each file contains:
        #   du_real_all / du_imag_all : ground-truth scattered wavefield (real/imag)
        #   u0_real_all / u0_imag_all : background (incident) wavefield (real/imag)
        #   v                         : velocity model
        #   shot_loc_list             : horizontal indices of source positions
        dict = sio.loadmat(f'../dataset/test/{md}_{freq}Hz.mat')
        du_real = dict['du_real_all']
        du_imag = dict['du_imag_all']
        u0_real = dict['u0_real_all']
        u0_imag = dict['u0_imag_all']
        vel = dict['v']
        vel = normalizer_vel(vel)                 # Normalize velocity to [0, 1]
        sx_loc_list = dict['shot_loc_list'][0]    # Source x-location indices (1-based)

        # ---------------------------------------------------------------------
        # 6b. Assemble complex wavefields as 2-channel tensors [real, imag]
        # ---------------------------------------------------------------------
        # Shape after stacking: (n_shots, 2, nz, nx)
        du = np.stack((du_real, du_imag), axis=1)
        u0 = np.stack((u0_real, u0_imag), axis=1)

        # Move tensors to GPU
        du = th.tensor(du, dtype=th.float32).to(device=device)
        u0 = th.tensor(u0, dtype=th.float32).to(device=device)
        # Add batch and channel dims: (1, 1, nz, nx) → replicate over all shots
        vel = th.tensor(vel, dtype=th.float32).unsqueeze(0).unsqueeze(1).to(device=device)

        # Number of shots and spatial dimensions
        b, _, w, h = u0.shape

        # Broadcast the single velocity model to match the batch of shots
        vel = vel.repeat(b, 1, 1, 1)   # Shape: (b, 1, nz, nx)

        # ---------------------------------------------------------------------
        # 6c. Run diffusion sampling
        # ---------------------------------------------------------------------
        # Returns:
        #   sample      : final predicted scattered wavefield, shape (b, 2, nz, nx)
        #   sample_all  : intermediate noisy samples at each denoising step
        #   pred_xstart : predicted x_0 at each denoising step (DDIM posterior mean)
        #   pde_loss_before/after : Helmholtz residual before and after guidance step
        sample, sample_all, pred_xstart, pde_loss_before, pde_loss_after = sample_fn(
            model, u0, vel,
            (b, args.out_channels, w, h),
            dh=args.dh,                       # Grid spacing (km), used in PDE residual
            omega=omega,                      # Angular frequency for Helmholtz guidance
            scale_factor=args.scale_factor,   # PDE guidance strength coefficient
            clip_denoised=args.clip_denoised, # Clip predicted x_0 to [-1, 1]
            model_kwargs=model_kwargs,
        )

        # Stack intermediate outputs along the time-step dimension for later analysis
        sample_all  = th.stack(sample_all)    # Shape: (T, b, 2, nz, nx)
        pred_xstart = th.stack(pred_xstart)   # Shape: (T, b, 2, nz, nx)

        # ---------------------------------------------------------------------
        # 6d. Source-point near-field correction
        # ---------------------------------------------------------------------
        # The scattered wavefield has a near-singularity directly at the source
        # location. Replace the source grid point with a 4-point average of its
        # immediate neighbors (cross stencil) to suppress the singular value.
        for index, sx_loc in enumerate(sx_loc_list):
            # Apply correction to the predicted scattered wavefield
            sample[index, :, 1, sx_loc-1] = 0.25 * (
                sample[index, :, 1, sx_loc-2] +   # left neighbor
                sample[index, :, 1, sx_loc]   +   # right neighbor
                sample[index, :, 0, sx_loc-1] +   # top neighbor
                sample[index, :, 2, sx_loc-1]     # bottom neighbor
            )
            # Apply the same correction to the ground-truth for a fair MSE comparison
            du[index, :, 1, sx_loc-1] = 0.25 * (
                du[index, :, 1, sx_loc-2] +
                du[index, :, 1, sx_loc]   +
                du[index, :, 0, sx_loc-1] +
                du[index, :, 2, sx_loc-1]
            )

        # ---------------------------------------------------------------------
        # 6e. Evaluate prediction accuracy
        # ---------------------------------------------------------------------
        # Compute MSE separately for the real and imaginary parts of the
        # scattered wavefield (lower is better)
        accs_real = criterion(sample[:, 0], du[:, 0])
        accs_imag = criterion(sample[:, 1], du[:, 1])

        # ---------------------------------------------------------------------
        # 6f. Save results to .mat file
        # ---------------------------------------------------------------------
        sio.savemat(
            f'{dir_output}{md}_{freq}Hz_out.mat',
            {
                'du_real_pred' : sample[:, 0].squeeze().cpu().numpy(),  # Predicted real part
                'du_imag_pred' : sample[:, 1].squeeze().cpu().numpy(),  # Predicted imaginary part
                'sample_all'   : sample_all.squeeze().cpu().numpy(),    # All intermediate denoising steps
                'pred_xstart'  : pred_xstart.squeeze().cpu().numpy(),   # Predicted x_0 at each step
                'accs_real'    : accs_real.item(),                      # MSE for real part
                'accs_imag'    : accs_imag.item(),                      # MSE for imaginary part
                'pde_loss_before': np.array(pde_loss_before, dtype=np.float32),  # Helmholtz residual before guidance
                'pde_loss_after' : np.array(pde_loss_after,  dtype=np.float32),  # Helmholtz residual after guidance
            }
        )

    logger.log("sampling complete")


def create_argparser():
    """
    Build the command-line argument parser and populate it with default values.

    Key arguments:
        clip_denoised   (bool)  : Whether to clip the predicted x_0 to [-1, 1]
                                  during each denoising step.
        use_ddim        (bool)  : If True, use DDIM deterministic sampling;
                                  otherwise use standard DDPM stochastic sampling.
        dh              (float) : Spatial grid spacing in km, required for
                                  computing the Helmholtz PDE residual in physical
                                  units during PDE-guided sampling.
        scale_factor    (float) : Strength coefficient for PDE gradient guidance.
                                  Controls how strongly the Helmholtz residual
                                  corrects the score function at each step.
                                  Active only when pde_guide=True.
        model_path      (str)   : Path to the pre-trained GNO checkpoint (.pt).
        timestep_respacing (str): Controls the DDIM step schedule.
                                  Set to "" during training (full DDPM chain).
                                  Set to "DDIMN" at inference to use N-1 steps,
                                  e.g., "DDIM50" for 49 denoising iterations.

    Returns:
        argparse.ArgumentParser: Fully configured argument parser.
    """
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        dh=0.025,           # Grid spacing: 25 m = 0.025 km
        scale_factor=0.002, # Default PDE guidance strength
        model_path="../../trained_GNO.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # Run inference sequentially over all target frequencies.
    # The diffusion model is re-initialized for each frequency since
    # the test data files and angular frequency (omega) differ per run.
    freq_list = [4, 8, 12, 15]
    for freq in freq_list:
        main(freq)