"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append('../')
import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt
from improved_diffusion.image_datasets import load_data
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from PIL import Image
from torchvision.utils import save_image
from datasets.generators import pendulum_script as pd
from improved_diffusion import metrics as mt
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.models as models
import torch.nn as nn
from improved_diffusion.nn import GaussianConvEncoderClf

fid = FrechetInceptionDistance(feature=64)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # LOAD DATASET
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        split="test"
    )

    logger.log("test...")
    logger.log("sampling...")
    model.to(dist_util.dev())
    model.eval() # EVALUATION MODE

    logger.log("sampling...")
    all_images = []
    all_labels = []

    # MORHPO
    all_images_thickness = []
    all_images_intensity = []

    # PENDULUM
    pend_scale = np.array([[2,42],[104,44],[7.5, 4.5],[11,8]])
    all_images_angle = []
    all_images_light = []
    all_images_shadlen = []
    all_images_shadowpos = []
    
    angle_distances = []
    light_distances = []
    shad_len_distances = []
    shad_pos_distances = []
    
    real_angle = []

    # CIRCUIT
    all_images_arm = []
    all_images_blue = []
    all_images_green = []
    all_images_red = []
    
    load_classifier = True
    if load_classifier:
        # clf = GaussianConvEncoderClf(in_channels=4, latent_dim=512, num_vars=4)
        # clf.load_state_dict(th.load('../results/pendulum/classifier/classifier_angle_best.pth'))
        # clf.eval()
        
        clf1 = GaussianConvEncoderClf(in_channels=4, latent_dim=512, num_vars=4)
        clf1.load_state_dict(th.load('../results/pendulum/classifier/classifier_angle_best.pth'))
        clf1.eval()
        
        clf2 = GaussianConvEncoderClf(in_channels=4, latent_dim=512, num_vars=4)
        clf2.load_state_dict(th.load('../results/pendulum/classifier/classifier_light_best.pth'))
        clf2.eval()
        
        clf3 = GaussianConvEncoderClf(in_channels=4, latent_dim=512, num_vars=4)
        clf3.load_state_dict(th.load('../results/pendulum/classifier/classifier_shadowlen_best.pth'))
        clf3.eval()
        
        clf4 = GaussianConvEncoderClf(in_channels=4, latent_dim=512, num_vars=4)
        clf4.load_state_dict(th.load('../results/pendulum/classifier/classifier_shadowpos_best.pth'))
        clf4.eval()
        # exit(0)

    count = 0
    while len(all_images) * args.batch_size < args.num_samples:

        batch, cond = next(data)
        count += 1

        noise = th.randn_like(batch).to(dist_util.dev())
        t = th.ones((batch.shape[0]), dtype=th.int64) * 249
        t = t.to(dist_util.dev())

        x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)
        
        if "morphomnist" in args.data_dir:
            cond["c"] = cond["c"].to(dist_util.dev())
            cond["y"] = cond["y"].to(dist_util.dev())
            # c = cond["c"].to(dist_util.dev())

            c = cond
            c["c"][:, 0] = -0.2
            # c[:, 0] = 0.5 


            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            
            sample = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images_thickness.extend([sample.cpu().numpy() for sample in gathered_samples])


            c = cond
            c["c"][:, 1] = -0.7

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images_intensity.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        elif "pendulum" in args.data_dir:
            
            
            # UNIFORMLY SAMPLE A RANDOM INTERVENTION VALUE
            # angle_int = np.random.uniform(-40, 44)
            angle_int = np.random.uniform(60, 148)
            # angle_int = np.random.uniform(3, 9)
            # angle_int = np.random.uniform(3, 15)

            # angle_norm = (angle_int - pend_scale[0][0]) / (pend_scale[0][1] - 0)
            angle_norm = (cond["c"][:, 0] - pend_scale[0][0]) / (pend_scale[0][1] - 0)
            light_norm = (angle_int - pend_scale[1][0]) / (pend_scale[1][1] - 0)
            len_norm = (cond["c"][:, 2] - pend_scale[2][0]) / (pend_scale[2][1] - 0)
            pos_norm = (cond["c"][:, 3] - pend_scale[3][0]) / (pend_scale[3][1] - 0)

            # angle_norm = angle_int
            # light_norm = cond["c"][:, 1]
            # len_norm = cond["c"][:, 2]
            # pos_norm = cond["c"][:, 3]
            
            labels = cond['c'].clone()
            print(labels.shape)
            exit(0)
            # print(labels[0])
            labels[:, 1] = th.tensor(angle_int)
            
            # print(labels[:, 0].shape)
            # exit(0)
            
            # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES
            X_real, v = pd.generate(labels[:, 0], labels[:, 1])

            print(v.shape)
            exit(0)
            
            angle_value = (v[:, 0] - pend_scale[0][0]) / (pend_scale[0][1] - 0)
            light_value = (v[:, 1] - pend_scale[1][0]) / (pend_scale[1][1] - 0)
            shadlen_value = (v[:, 2] - pend_scale[2][0]) / (pend_scale[2][1] - 0)
            shadpos_value = (v[:, 3] - pend_scale[3][0]) / (pend_scale[3][1] - 0)
            
            # c = cond

            # c["c"][:, 0] = angle_norm
            # c["c"][:, 1] = light_norm
            # c["c"][:, 2] = len_norm
            # c["c"][:, 3] = pos_norm
            
            cond["c"][:, 0] = angle_norm
            cond["c"][:, 1] = light_norm
            cond["c"][:, 2] = len_norm
            cond["c"][:, 3] = pos_norm

            cond["c"] = cond["c"].to(dist_util.dev())
            # c["c"] = c["c"].to(dist_util.dev())
            
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            
            noise = th.randn_like(batch).to(dist_util.dev())
            t = th.ones((batch.shape[0]), dtype=th.int64) * 249
            t = t.to(dist_util.dev())

            x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)
            
            sample = sample_fn(
                model,
                (args.batch_size, 4, args.image_size, args.image_size),
                #noise=x_t,
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,
            )
            
            
            out = clf1(th.tensor(sample.cpu()))
            angle_distances.append(nn.L1Loss()(out, th.tensor(angle_value).unsqueeze(1)))
            
            out = clf2(th.tensor(sample.cpu()))
            light_distances.append(nn.L1Loss()(out, th.tensor(light_value).unsqueeze(1)))
            
            out = clf3(th.tensor(sample.cpu()))
            shad_len_distances.append(nn.L1Loss()(out, th.tensor(shadlen_value).unsqueeze(1)))
            
            out = clf4(th.tensor(sample.cpu()))
            shad_pos_distances.append(nn.L1Loss()(out, th.tensor(shadpos_value).unsqueeze(1)))
            
            
            
            # cond["c"] = cond["c"].to(dist_util.dev())

            # # PENDULUM ANGLE
            # angle_int = np.random.uniform(-40, 44)
            # # angle_int = np.random.uniform(60, 148)
            # # angle_int = np.random.uniform(3, 15)
            
            # angle_norm = (angle_int - pend_scale[0][0]) / (pend_scale[0][1] - 0)
            # light_norm = (cond["c"][:, 1] - pend_scale[1][0]) / (pend_scale[1][1] - 0)
            # len_norm = (cond["c"][:, 2] - pend_scale[2][0]) / (pend_scale[2][1] - 0)
            # pos_norm = (cond["c"][:, 3] - pend_scale[3][0]) / (pend_scale[3][1] - 0)
            
            # labels = cond['c'].clone().cpu()
            # # print(labels[0])
            # labels[:, 0] = th.tensor(angle_int)
            
            # # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES
            # X_real, v = pd.generate(labels[:, 0], labels[:, 1])
            # real_angle.append(th.tensor(X_real))
                
            # c = cond
            # c["c"][:, 0] = angle_norm
            # c["c"][:, 1] = light_norm
            # c["c"][:, 2] = len_norm
            # c["c"][:, 3] = pos_norm
            

            # sample_fn = (
            #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            # )
            
            # sample = sample_fn(
            #     model,
            #     (args.batch_size, 4, args.image_size, args.image_size),
            #     #noise=x_t,
            #     clip_denoised=args.clip_denoised,
            #     model_kwargs=cond,
            # )

            # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            # all_images_angle.extend([np.rint(sample.cpu().numpy()) for sample in gathered_samples])
            
            # save_image(batch[:16], '../results/pendulum/label_conditional/cfid/original.png')
                
            # # SAVE PENDULUM ANGLE INTERVENED IMAGE
            # arr = np.concatenate(all_images_angle, axis=0)
            # arr = arr[: args.num_samples]
            # temp = arr[:16]

            # temp = th.tensor(temp, dtype=th.float32)

            # # save_image(th.tensor(all_images_angle[:16], dtype=th.float32), '../results/pendulum/causaldiffae/cfid/generated_angle.png')
            # save_image(temp, '../results/pendulum/label_conditional/cfid/generated_angle.png')

            # save_image(real_angle[0][:16], '../results/pendulum/label_conditional/cfid/true_angle.png')

            # exit(0)

            # # LIGHT POSITION
            # c = cond
            # c["c"][:, 1] = -0.32

            # sample_fn = (
            #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            # )
            # sample = sample_fn(
            #     model,
            #     (args.batch_size, 4, args.image_size, args.image_size),
            #     #noise=x_t,
            #     clip_denoised=args.clip_denoised,
            #     model_kwargs=cond,
            # )

            # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            # all_images_light.extend([sample.cpu().numpy() for sample in gathered_samples])
            
            # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            # all_images_angle.extend([np.rint(sample.cpu().numpy()) for sample in gathered_samples])


            # # SHADOW LENGTH
            # c = cond
            # c["c"][:, 2] = 0.7

            # sample_fn = (
            #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            # )
            # sample = sample_fn(
            #     model,
            #     (args.batch_size, 4, args.image_size, args.image_size),
            #     #noise=x_t,
            #     clip_denoised=args.clip_denoised,
            #     model_kwargs=cond,
            # )

            # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            # all_images_shadlen.extend([sample.cpu().numpy() for sample in gathered_samples])


            # # SHADOW POSITION
            # c = cond
            # c["c"][:, 3] = -0.4

            # sample_fn = (
            #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            # )
            # sample = sample_fn(
            #     model,
            #     (args.batch_size, 4, args.image_size, args.image_size),
            #     #noise=x_t,
            #     clip_denoised=args.clip_denoised,
            #     model_kwargs=cond,
            # )

            # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            # all_images_shadowpos.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        elif "circuit" in args.data_dir:
            cond["c"] = cond["c"].to(dist_util.dev())

            # PENDULUM ANGLE
            c = cond
            c["c"][:, 0] = 0.3

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                #noise=x_t,
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images_arm.extend([sample.cpu().numpy() for sample in gathered_samples])

            # LIGHT POSITION
            c = cond
            c["c"][:, 1] = 0.9

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                #noise=x_t,
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images_blue.extend([sample.cpu().numpy() for sample in gathered_samples])

            # SHADOW LENGTH
            c = cond
            c["c"][:, 2] = 0.9

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                #noise=x_t,
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images_green.extend([sample.cpu().numpy() for sample in gathered_samples])


            # SHADOW POSITION
            c = cond
            c["c"][:, 3] = 0.9

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                #noise=x_t,
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images_red.extend([sample.cpu().numpy() for sample in gathered_samples])

        break

    
    
    
    generate_intervention = False
    
    if generate_intervention == True:
        if "morphomnist" in args.data_dir:
            arr = np.concatenate(all_images_thickness, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[:16]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/morphomnist/labels_conditional/morphomnist_cond_thickness.png')

            arr = np.concatenate(all_images_intensity, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[:16]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/morphomnist/labels_conditional/morphomnist_cond_intensity.png')
        elif "pendulum" in args.data_dir:
            save_image(batch[0], '../results/pendulum/label_conditional/original.png')

            arr = np.concatenate(all_images_angle, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[0]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/pendulum/label_conditional/pendulum_cond_angle.png')


            arr = np.concatenate(all_images_light, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[0]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/pendulum/label_conditional/pendulum_cond_light.png')


            arr = np.concatenate(all_images_shadlen, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[0]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/pendulum/label_conditional/pendulum_cond_shadlen.png')


            arr = np.concatenate(all_images_shadowpos, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[0]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/pendulum/label_conditional/pendulum_cond_shadpos.png')
        
        elif "circuit" in args.data_dir:
            save_image(batch[7], '../results/circuit/label_conditional/original.png')

            arr = np.concatenate(all_images_arm, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[7]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/circuit/label_conditional/robot_arm.png')


            arr = np.concatenate(all_images_blue, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[7]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/circuit/label_conditional/blue_light.png')


            arr = np.concatenate(all_images_green, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[7]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/circuit/label_conditional/green_light.png')


            arr = np.concatenate(all_images_red, axis=0)
            arr = arr[: args.num_samples]
            temp = arr[7]

            temp = th.tensor(temp, dtype=th.float32)
            save_image(temp, '../results/circuit/label_conditional/red_light.png')
    else:
        mean_dist = th.tensor(sum(angle_distances) / len(angle_distances))
        gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
        print(f"Angle MAE: {sum(gathered_samples) / len(gathered_samples)}")
        
        mean_dist = th.tensor(sum(light_distances) / len(light_distances))
        gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
        print(f"Light MAE: {sum(gathered_samples) / len(gathered_samples)}")
        
        mean_dist = th.tensor(sum(shad_len_distances) / len(shad_len_distances))
        gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
        print(f"Shad Len MAE: {sum(gathered_samples) / len(gathered_samples)}")
        
        mean_dist = th.tensor(sum(shad_pos_distances) / len(shad_pos_distances))
        gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
        print(f"Shad Pos MAE: {sum(gathered_samples) / len(gathered_samples)}")
    
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        data_dir="",
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
