# -----------------------------------------------------------------------------
# Title: Training GNO
# Description:
#     This script trains a GNO to represent the scattered wavefield. 
#     The training process involves:
#     - Loading training data from a specified directory.
#     - Creating a GNO model with diffusion processes.
#     - Utilizing a schedule sampler for data sampling.
#     - Training the model with configurations provided as arguments.
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
    Main function to train the GNO model.
    Includes data loading, model initialization, and the training loop.
    """
    # Parse command-line arguments
    args = create_argparser().parse_args()

    # Configure logger for tracking training progress
    logger.configure()

    # Set device to GPU
    device = th.device('cuda')

    # Initialize the model and diffusion process
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Move the model to the GPU
    model.to(device)

    # Create a schedule sampler for data sampling (uniform or loss-based)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    # Load training data
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,        # Directory containing training data
        batch_size=args.batch_size,    # Batch size for training
        class_cond=args.class_cond,    # Whether to use class-conditional sampling
    )

    # Start the training loop
    logger.log("training...")
    TrainLoop(
        model=model,                  # The GNO model
        diffusion=diffusion,          # The diffusion process
        data=data,                    # Data loader for training data
        batch_size=args.batch_size,   # Batch size for training
        microbatch=args.microbatch,   # Microbatch size (for gradient accumulation)
        lr=args.lr,                   # Learning rate for optimization
        ema_rate=args.ema_rate,       # Exponential moving average rate
        log_interval=args.log_interval,  # Interval for logging training progress
        save_interval=args.save_interval,  # Interval for saving model checkpoints
        resume_checkpoint=args.resume_checkpoint,  # Checkpoint to resume from, if any
        use_fp16=args.use_fp16,       # Whether to use mixed precision (FP16)
        fp16_scale_growth=args.fp16_scale_growth,  # Scaling factor for FP16 training
        schedule_sampler=schedule_sampler,  # Sampling strategy for training
        weight_decay=args.weight_decay,     # Weight decay for regularization
        lr_anneal_steps=args.lr_anneal_steps,  # Steps for learning rate annealing
    ).run_loop()  # Start the main training loop


def create_argparser():
    """
    Create an argument parser to configure hyperparameters and model settings.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    defaults = dict(
        data_dir="../dataset/train/",        # Path to the synthetic data directory
        schedule_sampler="uniform",          # Sampling strategy (uniform by default)
        lr=1e-4,                             # Learning rate
        weight_decay=5e-5,                   # Weight decay for regularization
        lr_anneal_steps=600000,              # Steps for learning rate annealing
        batch_size=64,                       # Batch size for training
        microbatch=-1,                       # -1 disables microbatches
        ema_rate="0.999",                    # comma-separated list of EMA values
        log_interval=100,                    # Interval for logging progress
        save_interval=10000,                 # Interval for saving model checkpoints
        resume_checkpoint="",                # Path to checkpoint to resume training
        use_fp16=False,                      # Whether to use mixed precision (FP16)
        fp16_scale_growth=1e-3,              # Scaling factor for FP16 training
    )
    defaults.update(model_and_diffusion_defaults())  # Merge defaults with model settings
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # Automate adding arguments to parser
    return parser


if __name__ == "__main__":
    main()
