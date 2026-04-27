# -----------------------------------------------------------------------------
# Title: Seismic scattered wavefield representation with GNO
# Description:
#     This script performs sampling on test data using a trained GNO.
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

def main():
    """
    Main function to perform sampling on test data
    using a trained GNO. Supports both DDPM and DDIM sampling methods.
    """
    # Parse command-line arguments
    args = create_argparser().parse_args()

    # Set device to GPU
    device = th.device('cuda')

    # Configure logger for tracking progress and saving results
    logger.configure()

    # Create model and diffusion process
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(f'{args.model_path}', map_location=device)
    )
    model.to(device=device)
    model.eval()

    # Define the mean squared error loss function
    criterion = th.nn.MSELoss()
    logger.log("sampling...")

    # Number of data samples to process
    data_num = 8
 
    # Test data frequency
    freq = 12
    omega = 2 * math.pi * freq
    
    # Test data smooth level
    smooth_level = 0

    # Set output directory based on the sampling method (DDPM or DDIM)
    if not args.use_ddim:
        if args.pde_guide:
            dir_output = f'./output_{freq}Hz/ddpm/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            dir_output = f'./output_{freq}Hz/ddpm/usepde_{args.pde_guide}/'
    else:
        if args.pde_guide:
            dir_output = f'./output_{freq}Hz/{args.timestep_respacing}/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            dir_output = f'./output_{freq}Hz/{args.timestep_respacing}/usepde_{args.pde_guide}/'
    os.makedirs(dir_output, exist_ok=True)

    # Placeholder for additional model parameters
    model_kwargs = {}

    # Select the sampling function (DDPM or DDIM)
    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    # Loop through each data sample for deterministic sampling
    for id in range(data_num):
        print(f'Sampling for test data {id+1}.mat with sigma {smooth_level}')

      # Load the current test sample from a .mat file
        dict = sio.loadmat(f'../dataset/test/data{freq}Hz/data{id+1}_sigma{smooth_level}.mat')
        du_real = dict['du_real_all']
        du_imag = dict['du_imag_all']
        u0_real = dict['u0_real_all']
        u0_imag = dict['u0_imag_all']  
        vel = dict['v']
        vel = normalizer_vel(vel)

        du = np.stack((du_real, du_imag), axis=1)
        u0 = np.stack((u0_real, u0_imag), axis=1)
        du = th.tensor(du, dtype = th.float32).to(device=device)
        u0 = th.tensor(u0, dtype = th.float32).to(device=device)
        vel = th.tensor(vel, dtype = th.float32).unsqueeze(0).unsqueeze(1).to(device=device)

        # Get input tensor dimensions
        b, _, w, h = u0.shape

        vel = vel.repeat(b, 1, 1, 1)

        # Perform sampling
        sample, sample_all, pred_xstart, pde_loss_before, pde_loss_after = sample_fn(
                model, u0, vel,
                (b, args.out_channels, w, h),
                dh=args.dh,
                omega=omega,
                scale_factor=args.scale_factor,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
        )

        # Intermediate output of the sampling process
        sample_all = th.stack(sample_all)
        pred_xstart = th.stack(pred_xstart)

        # Compute the MSE between the sampled result and the ground truth
        accs_real = criterion(sample[:, 0], du[:, 0])
        accs_imag = criterion(sample[:, 1], du[:, 1])

        # Save the sampled output and accuracy metrics to a .mat file
        sio.savemat(f'{dir_output}data{id+1}_sigma{smooth_level}_out.mat', 
                {'du_real_pred': sample[:, 0].squeeze().cpu().numpy(), 
                 'du_imag_pred': sample[:, 1].squeeze().cpu().numpy(),
                 'sample_all': sample_all.squeeze().cpu().numpy(),
                 'pred_xstart': pred_xstart.squeeze().cpu().numpy(),
                 'accs_real': accs_real.item(), 'accs_imag': accs_imag.item(), 
                 'pde_loss_before': np.array(pde_loss_before, dtype=np.float32),
                 'pde_loss_after': np.array(pde_loss_after, dtype=np.float32)})


    logger.log("sampling complete")


def create_argparser():
    """
    Create a command-line argument parser and add default and model-related arguments.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        dh=0.025,
        scale_factor=0.001,
        model_path="./checkpoints/trained_GNO.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
