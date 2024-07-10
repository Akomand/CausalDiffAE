import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel

# NUMBER OF CLASSES
NUM_CLASSES = 10
CONTEXT_DIM = 2
REP_DIM = 512


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        context_cond=False,
        rep_cond=False
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
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
    context_cond,
    rep_cond
):
    # UNET MODEL
    model = create_model(
        image_size,     # SIZE OF IMAGE
        num_channels,   # INPUT CHANNELS
        num_res_blocks, # HOW MANY RESIDUAL BLOCKS TO USE
        learn_sigma=learn_sigma, # LEARN VARIANCE INSTEAD OF FIXED VARIANCE
        class_cond=class_cond, # YES/NO FOR CLASS CONDITIONING (CONTEXT)
        use_checkpoint=use_checkpoint, # OPTIMIZATION TECHNIQUE -> DURING FORWARD PASS, DON'T STORE ACTIVATIONS AT EACH LAYER (BUT NEED TO RECOMPUTE) TRADE OFF TIME FOR MEMORY
        attention_resolutions=attention_resolutions, # VAT TYPE ATTENTION (ALL TOKEN CROSS ATTENTION) -> AT EACH LEVEL (8 CHANNEL, 16 CHANNEL, ETC)
        num_heads=num_heads, # HOW MANY ATTENTION HEADS
        num_heads_upsample=num_heads_upsample, #
        use_scale_shift_norm=use_scale_shift_norm, # HOW TO CONDITION IMAGES ON TIMESTEP
        dropout=dropout, # DROPOUT REGULARIZATION
        context_cond=context_cond,
        rep_cond=rep_cond
    )

    # DIFFUSION MODEL (THIS HAS FUNCTIONS TO PERFORM FORWARD AND REVERSE DIFFUSION)
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,      # HOW MANY TIMSTEPS
        learn_sigma=learn_sigma,    # LEARN VARIANCE
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,  # LINEAR NOISE SCHEDULE (FORWARD)
        use_kl=use_kl, # IF TRUE, L_VLB, ELSE L_HYBRID
        predict_xstart=predict_xstart,  # NOT PREDICTING THE ORIGINAL IMAGE (PREDICTING NOSIE)
        rescale_timesteps=rescale_timesteps,  # RESCALE TIMESTEP
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

# THIS IS WHERE UNET MODEL IS CREATED
def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    context_cond,
    rep_cond
):
    # UNET SPEC FOR DIFFERENT IMAGE SIZES
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4) # AT EACH CONV BLOCK, MULTIPLY BY THESE FACTORS
    elif image_size == 128:
        channel_mult = (1, 1, 2, 2, 4, 4) # AT EACH CONV BLOCK, MULTIPLY BY THESE FACTORS
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = [] # CONVERT ATTENTION RESOLUTIONS INTO NUMBER OF DOWNSAMPLING LAYERS UNTIL ATTENTION LAYERS INCLUDED
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # SPECIFY UNET MODEL (CHANGE THE CHANNELS AND PREDICTION OUTPUT CHANNELS BASED ON NUM CHANNELS IN INPUT IMAGE)
    return UNetModel(
        in_channels=1,
        model_channels=num_channels,
        out_channels=(1 if not learn_sigma else 2), # FIRST 3 CHANNEL PREDICT EPSILON AND SECOND 3 CHANNELS PREDICT VARIANCE
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        c_dim=(CONTEXT_DIM if context_cond else None),
        rep_dim=(REP_DIM if rep_cond else None),
        use_checkpoint=use_checkpoint,  # FALSE
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

    # return UNetModel(
    #     in_channels=3,
    #     model_channels=num_channels,
    #     out_channels=(3 if not learn_sigma else 6), # FIRST 3 CHANNEL PREDICT EPSILON AND SECOND 3 CHANNELS PREDICT VARIANCE
    #     num_res_blocks=num_res_blocks,
    #     attention_resolutions=tuple(attention_ds),
    #     dropout=dropout,
    #     channel_mult=channel_mult,
    #     num_classes=(NUM_CLASSES if class_cond else None),
    #     use_checkpoint=use_checkpoint,  # FALSE
    #     num_heads=num_heads,
    #     num_heads_upsample=num_heads_upsample,
    #     use_scale_shift_norm=use_scale_shift_norm,
    # )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
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
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
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
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
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
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
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
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps) # GET BETA VARIANCE SCHEDULE

    # TYPE OF LOSS
    if use_kl: # L_VLB
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas: # L_RESCALED_VLB
        loss_type = gd.LossType.RESCALED_MSE
    else: # L_HYBRID
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    # DURING SAMPLING (KIND OF LIKE DDIM) -> HOW TO SAMPLE (SPACED)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),  # [0, ..., T] FOR NO SPACING, O.W. [0, 6, 10, ..., T]
        betas=betas, # NOISE
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ), # PREDICING EPSILON INSTEAD OF PREDICTING ORIGINAL IMAGE (EMPIRICALLY BETTER)
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
