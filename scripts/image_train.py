"""
Train a diffusion model on images.
"""

import sys
sys.path.append('../')
import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

# START
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    # logger.configure(dir = "../results/morphomnist")
    # logger.configure(dir = "../results/morphomnist/causaldiffae_masked_p=0.8")
    # logger.configure(dir = "../results/pendulum/causaldiffae_masked")
    # logger.configure(dir = "../results/pendulum/label_conditional")
    # logger.configure(dir = "../results/pendulum/diffae_aligned")
    # logger.configure(dir = "../results/pendulum/causaldiffae")
    
    logger.configure(dir = "../results/circuit/causaldiffae_masked")
    # logger.configure(dir = "../results/circuit/diffae_unaligned")
    # logger.configure(dir = "../results/circuit/diffae")
    # logger.configure(dir = "../results/circuit/label_conditional")
    # logger.configure(dir = "../results/morphomnist/diffae_unaligned")
    # logger.configure(dir = "../results/pendulum/diffae_unaligned")

    logger.log("creating model and diffusion...")
    # CREATE MODEL (model is the UNET and diffusion is forward/reverse process and variance scheduling)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    # TIMESTEP SAMPLING (UNIFORM OR IMPORTANCE SAMPLING)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # LOAD DATASET
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        rep_cond=args.rep_cond,
        n_vars=args.n_vars,
        causal_modeling=args.causal_modeling,
        flow_based=args.flow_based,
        in_channels=args.in_channels,
        masking=args.masking
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        rep_cond=False,
        n_vars=None,
        causal_modeling=False,
        flow_based=False,
        in_channels=3,
        masking=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
