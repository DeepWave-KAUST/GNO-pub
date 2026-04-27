import argparse
import inspect
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel

# Total number of benchmark velocity model families used for class-conditional training
NUM_CLASSES = 4


def model_and_diffusion_defaults():
    """
    Return the default hyperparameters for both the GNO (UNet) architecture
    and the Gaussian diffusion process.

    Channel layout for in_channels=5:
        The UNet receives the concatenation of the noisy scattered wavefield
        and the conditioning signals along the channel dimension:
            - du_noisy : 2 channels (real + imaginary parts of the noisy target)
            - u0       : 2 channels (real + imaginary parts of the background wavefield)
            - vel      : 1 channel  (normalized velocity model)
        Total: 2 + 2 + 1 = 5 input channels.

    Returns:
        dict: Default configuration dictionary consumed by create_model_and_diffusion().
    """
    return dict(
        # ----- UNet architecture -----
        in_channels=5,               # Noisy du (2) + background wavefield u0 (2) + velocity (1)
        num_channels=64,             # Base feature-map width; multiplied at each stage by channel_mult
        out_channels=2,              # Predicted scattered wavefield: real and imaginary parts
        channel_mult=(1, 2, 4, 8, 16),   # Per-stage channel multipliers for the UNet encoder/decoder
        num_res_blocks=2,            # Number of residual blocks per resolution stage
        num_heads=4,                 # Number of self-attention heads in attention blocks
        num_heads_upsample=-1,       # Attention heads in the decoder; -1 mirrors the encoder value
        attention_resolutions=(4, 8, 16),  # Spatial resolutions (in grid cells) where attention is applied
        dropout=0.0,                 # Dropout probability; 0.0 disables dropout
        use_checkpoint=False,        # Gradient checkpointing to trade compute for memory
        use_scale_shift_norm=True,   # AdaGN-style scale-and-shift conditioning in residual blocks

        # ----- Variance / loss options -----
        learn_sigma=False,           # If True, the network also predicts the diffusion variance
        sigma_small=False,           # If learn_sigma=False, use FIXED_SMALL instead of FIXED_LARGE variance
        use_kl=False,                # Use rescaled KL loss; requires learn_sigma=True
        rescale_learned_sigmas=False,# Rescale the learned-sigma MSE loss term (IDDPM Eq. 18)

        # ----- Diffusion schedule -----
        diffusion_steps=1000,        # Total number of forward diffusion steps T
        noise_schedule="cosine",     # Beta schedule type: "linear" or "cosine" (recommended)
        predict_xstart=True,         # Predict x_0 directly rather than the noise epsilon
        rescale_timesteps=True,      # Rescale t to [0, 1000] regardless of actual T

        # ----- Timestep respacing (DDIM) -----
        # IMPORTANT: must be "" (empty string) during training so the full T-step
        # chain is used. At inference, set to "DDIMN" (e.g., "DDIM50") to use
        # N-1 denoising steps. The default "ddim2" here is only suitable for
        # quick debugging — override this via command line for real experiments.
        timestep_respacing="ddim2",

        # ----- Conditioning -----
        class_cond=False,            # Class-conditional training using NUM_CLASSES labels
                                     # (set False for wavefield/velocity conditioning only)

        # ----- PDE guidance -----
        pde_guide=True,              # Apply Helmholtz PDE gradient guidance during sampling.
                                     # Has no effect during training; active only in sample.py.
                                     # Set to False for unconditional (data-driven only) sampling.
        fd_order=4,                  # Finite-difference stencil order for PDE residual computation
                                     # (4th-order recommended for accuracy at coarse grid spacings)
    )


def create_model_and_diffusion(
    class_cond,
    learn_sigma,
    sigma_small,
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    pde_guide,
    fd_order,
):
    """
    Instantiate and return both the GNO denoising network and the diffusion process.

    This is the single entry point used by both train.py and sample.py to
    construct the model/diffusion pair from a flat configuration dictionary.

    Args:
        class_cond            (bool) : Enable class-conditional UNet (uses NUM_CLASSES).
        learn_sigma           (bool) : Network also outputs the diffusion variance.
        sigma_small           (bool) : Use FIXED_SMALL variance when learn_sigma=False.
        in_channels           (int)  : UNet input channels (du_noisy + u0 + vel = 5).
        num_channels          (int)  : Base UNet feature-map width.
        out_channels          (int)  : UNet output channels (real + imag = 2).
        channel_mult          (tuple): Per-stage channel multipliers.
        num_res_blocks        (int)  : Residual blocks per resolution stage.
        num_heads             (int)  : Attention heads in encoder.
        num_heads_upsample    (int)  : Attention heads in decoder (-1 = same as encoder).
        attention_resolutions (tuple): Grid resolutions where attention is applied.
        dropout               (float): Dropout probability.
        diffusion_steps       (int)  : Total forward diffusion steps T.
        noise_schedule        (str)  : Beta schedule type ("linear" or "cosine").
        timestep_respacing    (str)  : "" for training; "DDIMN" for DDIM inference.
        use_kl                (bool) : Use rescaled KL divergence loss.
        predict_xstart        (bool) : Predict x_0 (True) or epsilon (False).
        rescale_timesteps     (bool) : Rescale t to [0, 1000].
        rescale_learned_sigmas(bool) : Rescale learned-sigma MSE loss term.
        use_checkpoint        (bool) : Gradient checkpointing to reduce memory.
        use_scale_shift_norm  (bool) : AdaGN scale-and-shift in residual blocks.
        pde_guide             (bool) : Enable Helmholtz PDE guidance at sampling.
        fd_order              (int)  : FD stencil order for PDE residual.

    Returns:
        tuple: (model, diffusion)
            model     : UNetModel — the GNO denoising network.
            diffusion : SpacedDiffusion — the (possibly respaced) diffusion process.
    """
    model = create_model(
        in_channels=in_channels,
        num_channels=num_channels,
        out_channels=out_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        pde_guide=pde_guide,
        fd_order=fd_order,
    )
    return model, diffusion


def create_model(
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    Instantiate the GNO denoising UNet.

    When learn_sigma=True, the UNet outputs 2*out_channels predictions:
    the first half is the denoised signal and the second half is the
    log-variance used to parameterize the diffusion variance.

    Args:
        (see create_model_and_diffusion for argument descriptions)

    Returns:
        UNetModel: The configured denoising network.
    """
    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        # When learn_sigma=True, double output channels to also predict variance
        out_channels=(out_channels * 2 if learn_sigma else out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        # Pass number of classes only when class-conditional; None = unconditional
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    pde_guide=False,
    fd_order=4,
):
    """
    Build and return the (possibly timestep-respaced) Gaussian diffusion process.

    The function selects the appropriate loss type, variance parameterization,
    and mean prediction target based on the supplied flags, then wraps everything
    in a SpacedDiffusion that supports both DDPM and DDIM sampling.

    Loss type priority (mutually exclusive):
        use_kl=True               → RESCALED_KL   (requires learn_sigma=True)
        rescale_learned_sigmas=True → RESCALED_MSE
        otherwise                 → MSE            (standard training objective)

    Mean prediction target:
        predict_xstart=True  → ModelMeanType.START_X  (predict x_0 directly)
        predict_xstart=False → ModelMeanType.EPSILON   (predict added noise)

    Variance type:
        learn_sigma=True    → LEARNED_RANGE  (network outputs log-variance interpolation)
        learn_sigma=False, sigma_small=False → FIXED_LARGE  (β_t, more stable early training)
        learn_sigma=False, sigma_small=True  → FIXED_SMALL  (β̃_t, lower variance posterior)

    Args:
        steps                 (int)  : Total forward diffusion steps T.
        learn_sigma           (bool) : Network also predicts diffusion variance.
        sigma_small           (bool) : Use β̃_t (smaller) when variance is fixed.
        noise_schedule        (str)  : Beta schedule ("linear" or "cosine").
        use_kl                (bool) : Use rescaled KL loss (requires learn_sigma=True).
        predict_xstart        (bool) : Predict x_0 instead of epsilon.
        rescale_timesteps     (bool) : Rescale t to [0, 1000] for schedule-agnostic training.
        rescale_learned_sigmas(bool) : Apply IDDPM rescaling to learned-sigma MSE term.
        timestep_respacing    (str)  : "" = full T-step DDPM chain (use during training).
                                       "DDIMN" = N-1 DDIM steps (use at inference).
        pde_guide             (bool) : Pass Helmholtz PDE guidance flag to SpacedDiffusion.
        fd_order              (int)  : Finite-difference order for PDE residual computation.

    Returns:
        SpacedDiffusion: Configured diffusion process supporting p_sample_loop
                         (DDPM) and ddim_sample_loop (DDIM).
    """
    # Compute the noise schedule betas for the full T-step chain
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # Select the training loss type
    if use_kl:
        # Variational lower bound loss; requires the network to predict variance
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        # Hybrid MSE + rescaled KL as in IDDPM (Eq. 18); balances signal and variance learning
        loss_type = gd.LossType.RESCALED_MSE
    else:
        # Standard MSE on the predicted signal (or noise); default for this project
        loss_type = gd.LossType.MSE

    # If no respacing is specified, use all T timesteps (standard DDPM training)
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        # Subset of timesteps selected by the respacing strategy
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        # Mean prediction: START_X (predict x_0) or EPSILON (predict added noise)
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        # Variance type: fixed (large or small) or learned by the network
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE   # β_t: slightly larger, recommended default
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL  # β̃_t: lower-variance posterior
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE    # Network outputs interpolation weights
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        pde_guide=pde_guide,
        fd_order=fd_order,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    Register all key-value pairs in default_dict as command-line arguments.

    Each key becomes a '--key' flag with its default value and inferred type.
    Boolean values are handled via str2bool to support "true"/"false" strings
    from the command line.

    Args:
        parser       (argparse.ArgumentParser): Parser to add arguments to.
        default_dict (dict): Mapping of argument names to their default values.
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str        # Treat None defaults as string arguments
        elif isinstance(v, bool):
            v_type = str2bool   # Handle bool flags as "true"/"false" strings
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Extract a subset of parsed arguments as a plain dictionary.

    Used to pass only the relevant keys from the full argparse namespace into
    create_model_and_diffusion(), avoiding unexpected keyword argument errors.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        keys (iterable of str)   : Keys to extract from args.

    Returns:
        dict: {key: getattr(args, key)} for each key in keys.
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Convert a string command-line argument to a Python bool.

    Accepts common truthy strings ("yes", "true", "t", "y", "1") and
    falsy strings ("no", "false", "f", "n", "0"), case-insensitively.
    Raises ArgumentTypeError for any other input.

    Reference:
        https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Args:
        v (bool or str): Value to convert.

    Returns:
        bool: Parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If v is not a recognizable boolean string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")