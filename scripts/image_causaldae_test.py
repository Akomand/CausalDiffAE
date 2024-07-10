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
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.nn import mean_flat, kl_normal, reparameterize
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from PIL import Image
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets.generators import pendulum_script as pd
from datasets.generators import morphomnist_script as ms
from improved_diffusion import metrics as mt
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.models as models
import torch.nn as nn
from improved_diffusion.nn import GaussianConvEncoderClf
from torch.distributions.gamma import Gamma

fid = FrechetInceptionDistance(feature=64)

def main():
    args = create_argparser().parse_args()
    # print(args.data_dir)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print(dist_util.dev())
    model.to(dist_util.dev())
    model.eval() # EVALUATION MODE
    
    # LOAD DATASET
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond
        # split="test"
    )

    test_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        split="test"
    )
    

    logger.log("test...")
    logger.log("sampling...")

    # FLAGS 
    eval_disentanglement = False
    traversal = False
    generate_interventions = True
    
    
    all_images = []
    all_labels = []

    mnist_scale = []
    all_images_thickness = []
    all_images_intensity = []

    thickness_distances = []
    intensity_distances = []
    

    pend_scale = np.array([[2,42],[104,44],[7.5, 4.5],[11,8]])
    mnist_scale = np.array([[3.4, 2.4], [161, 94]])

    real_angle = []
    real_light = []
    real_shadow_len = []
    real_shadow_pos = []
    
    all_images_angle = []
    all_images_light = []
    all_images_shadow_len = []
    all_images_shadow_pos = []
    
    angle_only = []
    light_only = []
    shadow_len_only = []
    shadow_pos_only = []

    all_images_arm = []
    all_images_blue = []
    all_images_green = []
    all_images_red = []
    
    reconstructions = []
    
    angle_distances = []
    light_distances = []
    shad_len_distances = []
    shad_pos_distances = []
    
    # distances = []
    w = None

    load_classifier = True
    if load_classifier:
        # clf = GaussianConvEncoderClf(in_channels=4, latent_dim=512, num_vars=4)
        # clf.load_state_dict(th.load('../results/pendulum/classifier/classifier_angle_best.pth'))
        # clf.eval()
        if "morphomnist" in args.data_dir:
            clf1 = GaussianConvEncoderClf(in_channels=1, latent_dim=512, num_vars=2)
            clf1.load_state_dict(th.load('../results/morphomnist/classifier/classifier_thickness_best.pth'))
            clf1.eval()

            clf2 = GaussianConvEncoderClf(in_channels=1, latent_dim=512, num_vars=2)
            clf2.load_state_dict(th.load('../results/morphomnist/classifier/classifier_intensity_best.pth'))
            clf2.eval()

        elif "pendulum" in args.data_dir:
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
        
    if eval_disentanglement:
        if "morphomnist" in args.data_dir:
            rep_train = np.empty((60000, 512))
            y_train = np.empty((60000, 2))

            batch_idx = 0
            while batch_idx < 3750:
                batch, cond = next(data)
                A = th.tensor([[0, 1], [0, 0]], dtype=th.float32).to(batch.device)
                # print(batch_idx)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)
                z = z.reshape(-1, 512)
                # print(z.shape)
                # exit(0)
                rep_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()
                # print(z.shape)
                y_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+ cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                batch_idx += 1
            

            rep_test = np.empty((10000, 512))
            y_test = np.empty((10000, 2))
            batch_idx = 0
            while batch_idx < 625:
                batch, cond = next(test_data)
                A = th.tensor([[0, 1], [0, 0]], dtype=th.float32).to(batch.device)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)

                z = z.reshape(-1, 512)
                rep_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()
                y_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+ cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                batch_idx += 1

            scores, importance_matrix, code_importance = mt._compute_dci(rep_train.T, y_train.T, rep_test.T, y_test.T)
            print(scores)
        
        elif "pendulum" in args.data_dir:
    
            rep_train = np.empty((5482, 64))
            y_train = np.empty((5482, 4))

            batch_idx = 0
            while batch_idx < 342:
                batch, cond = next(data)

                A = th.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=th.float32).to(batch.device)
                # print(batch_idx)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)
                z = z.reshape(-1, 64)
                # print(z.shape)
  
                rep_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()

                y_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                batch_idx += 1
            

            rep_test = np.empty((1826, 64))
            y_test = np.empty((1826, 4))
            batch_idx = 0
            
            while batch_idx < 114:
                batch, cond = next(test_data)
                A = th.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=th.float32).to(batch.device)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)
                
                z = z.reshape(-1, 64)
                rep_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()
                y_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+ cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                batch_idx += 1

            scores, importance_matrix, code_importance = mt._compute_dci(rep_train.T, y_train.T, rep_test.T, y_test.T)
            print(scores)
        
        elif "circuit" in args.data_dir:
            rep_train = np.empty((50000, 512))
            y_train = np.empty((50000, 4))

            batch_idx = 0
            while batch_idx < 3125:
                batch, cond = next(data)

                A = th.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=th.float32).to(batch.device)
                # print(batch_idx)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)
                z = z.reshape(-1, 512)
                # print(z.shape)
                # exit(0)
                rep_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()

                y_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                batch_idx += 1
            

            rep_test = np.empty((10000, 512))
            y_test = np.empty((10000, 4))
            batch_idx = 0
            
            while batch_idx < 625:
                batch, cond = next(test_data)
                A = th.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=th.float32).to(batch.device)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)
                
                z = z.reshape(-1, 512)
                rep_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()
                y_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+ cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                batch_idx += 1

            scores, importance_matrix, code_importance = mt._compute_dci(rep_train.T, y_train.T, rep_test.T, y_test.T)
            print(scores)
        



    else:
        gamma_dist = Gamma(th.tensor(10.), th.tensor(5.))
        count = 0
        while len(all_images) * args.batch_size < args.num_samples:

            batch, cond = next(data)

            # save_image([batch[0]], './new_test.png')
            # print(cond["c"][0])
            # print(cond["real"][0])
            # exit(0)
            count += 1
            # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
            if "morphomnist" in args.data_dir:
                A = th.tensor([[0, 1], [0, 0]], dtype=th.float32).to(batch.device)


                # EVALUTING EFFECTIVENESS CODE
                # MORPHOMNIST ANGLE INTERVENTIONS
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001
    

                # UNIFORMLY SAMPLE A RANDOM INTERVENTION VALUE
                # thickness = np.random.uniform(64, 250)

                # # angle_norm = (angle_int - pend_scale[0][0]) / (pend_scale[0][1] - 0)
                # t_norm = (thickness - mnist_scale[1][0]) / (mnist_scale[1][1] - 0)
                # labels = cond['c'].clone()
                # # print(labels[0])
                # labels[:, 1] = th.tensor(thickness)
                
                # # print(labels[:, 0].shape)
                # # exit(0)
                
                # # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES
                # v = ms.generate(thickness=labels[:, 0], intensity=labels[:, 1])
                # # real_angle.append(th.tensor(X_real))
                
                # thickness_value = (v[:, 0] - mnist_scale[0][0]) / (mnist_scale[0][1] - 0)
                # intensity_value = (v[:, 1] - mnist_scale[1][0]) / (mnist_scale[1][1] - 0)

                # # mu[:, :16] = th.ones((args.batch_size, 16)) * 0.67 # 30
                # # mu[:, 16:32] = th.ones((args.batch_size, 16)) * angle_norm

                # # mu[:, :256] = th.ones((args.batch_size, 256)) * t_norm
                
                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                # z_post[:, 256:512] = th.ones((args.batch_size, 256)) * t_norm
                

                # z = reparameterize(z_post, var)

                # noise = th.randn_like(batch).to(dist_util.dev())
                # t = th.ones((batch.shape[0]), dtype=th.int64) * 249
                # t = t.to(dist_util.dev())

                # x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)


                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())
                # cond["y"] = cond["y"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 1, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )
                
                
                # out = clf1(th.tensor(sample.cpu()))
                # thickness_distances.append(nn.L1Loss()(out, th.tensor(thickness_value).unsqueeze(1)))
                
                # out = clf2(th.tensor(sample.cpu()))
                # intensity_distances.append(nn.L1Loss()(out, th.tensor(intensity_value).unsqueeze(1)))


                
                # THICKNESS INTERVENTIONS
                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                
                mu[:, :256] = th.ones((args.batch_size, 256)) * 0.2

                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)

                noise = th.randn_like(batch).to(dist_util.dev())
                t = th.ones((batch.shape[0]), dtype=th.int64) * 249
                t = t.to(dist_util.dev())

                x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)


                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                cond["z"] = z
                # cond["y"] = th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.int32) * 4
                cond["y"] = cond["y"].to(dist_util.dev())

                sample = sample_fn(
                    model,
                    (args.batch_size, 1, args.image_size, args.image_size),
                    noise=x_t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cond,
                    w=w
                )

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images_thickness.extend([sample.cpu().numpy() for sample in gathered_samples])
                
                

                # INTENSITY INTERVENTIONS
                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001

                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z_post[:, 256:] = th.ones((args.batch_size, 256)) * 0.2

                z = reparameterize(z_post, var)

                x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)


                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                cond["z"] = z
                # cond["y"] = th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.int32) * 4
                cond["y"] = cond["y"].to(dist_util.dev())

                sample = sample_fn(
                    model,
                    (args.batch_size, 1, args.image_size, args.image_size),
                    noise=x_t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cond,
                    w=w
                )

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images_intensity.extend([sample.cpu().numpy() for sample in gathered_samples])
            
            elif "pendulum" in args.data_dir:
                A = th.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=th.float32).to(batch.device)

                if traversal == True:
                    # RECONSTRUCTIONS
                    mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                    
                    var = th.ones(mu.shape).to(mu.device) * 0.001
                    
                    z_pre = model.causal_mask.causal_masking(mu, A)
                    z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)


                    z = reparameterize(z_post, var)

                    noise = th.randn_like(batch).to(dist_util.dev())
                    t = th.ones((batch.shape[0]), dtype=th.int64) * 249
                    t = t.to(dist_util.dev())
                    
                    x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)
                    
                    value = -0.5
                    for i in range(8):
                        # PENDULUM ANGLE INTERVENTIONS
                        mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                        var = th.ones(mu.shape).to(mu.device) * 0.001
                        mu[:, 16:32] = th.ones((args.batch_size, 16)) * value # 30
                        
                        z_pre = model.causal_mask.causal_masking(mu, A)
                        z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                        z = reparameterize(z_post, var)

                        sample_fn = (
                        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                    )
                        cond["z"] = z
                        cond["c"] = cond["c"].to(dist_util.dev())

                        sample = sample_fn(
                            model,
                            (args.batch_size, 4, args.image_size, args.image_size),
                            noise=x_t,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=cond,
                            w=w
                        )

                        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                        all_images_light.extend([sample.cpu().numpy() for sample in gathered_samples])
                        
                        value += 0.15



                # PENDULUM ANGLE INTERVENTIONS
                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                
                # UNIFORMLY SAMPLE A RANDOM INTERVENTION VALUE
                angle_int = np.random.uniform(-40, 44)
                # angle_int = np.random.uniform(60, 148)
                # angle_int = np.random.uniform(3, 9)
                # angle_int = np.random.uniform(3, 15)

                # angle_norm = (angle_int - pend_scale[0][0]) / (pend_scale[0][1] - 0)
                angle_norm = (angle_int - pend_scale[0][0]) / (pend_scale[0][1] - 0)
                labels = cond['c'].clone()
                # print(labels[0])
                labels[:, 0] = th.tensor(angle_int)
                
                # print(labels[:, 0].shape)
                # exit(0)
                
                # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES
                X_real, v = pd.generate(labels[:, 0], labels[:, 1])
                # real_angle.append(th.tensor(X_real))
                
                angle_value = (v[:, 0] - pend_scale[0][0]) / (pend_scale[0][1] - 0)
                light_value = (v[:, 1] - pend_scale[1][0]) / (pend_scale[1][1] - 0)
                shadlen_value = (v[:, 2] - pend_scale[2][0]) / (pend_scale[2][1] - 0)
                shadpos_value = (v[:, 3] - pend_scale[3][0]) / (pend_scale[3][1] - 0)

                # mu[:, :16] = th.ones((args.batch_size, 16)) * 0.67 # 30
                # mu[:, 16:32] = th.ones((args.batch_size, 16)) * angle_norm

                
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z_post[:, :16] = th.ones((args.batch_size, 16)) * angle_norm
                

                z = reparameterize(z_post, var)

                noise = th.randn_like(batch).to(dist_util.dev())
                t = th.ones((batch.shape[0]), dtype=th.int64) * 249
                t = t.to(dist_util.dev())

                x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)


                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                cond["z"] = z
                cond["c"] = cond["c"].to(dist_util.dev())

                sample = sample_fn(
                    model,
                    (args.batch_size, 4, args.image_size, args.image_size),
                    noise=x_t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cond,
                    w=w
                )
                
                
                out = clf1(th.tensor(sample.cpu()))
                angle_distances.append(nn.L1Loss()(out, th.tensor(angle_value).unsqueeze(1)))
                
                out = clf2(th.tensor(sample.cpu()))
                light_distances.append(nn.L1Loss()(out, th.tensor(light_value).unsqueeze(1)))
                
                out = clf3(th.tensor(sample.cpu()))
                shad_len_distances.append(nn.L1Loss()(out, th.tensor(shadlen_value).unsqueeze(1)))
                
                out = clf4(th.tensor(sample.cpu()))
                shad_pos_distances.append(nn.L1Loss()(out, th.tensor(shadpos_value).unsqueeze(1)))
                
                # print(len(distances))
                # exit(0)
            
                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_angle.extend([np.rint(sample.cpu().numpy()) for sample in gathered_samples])
                # all_images_angle.extend([sample.cpu().numpy() for sample in gathered_samples])
                
                
                
                

                # noise = th.randn_like(batch).to(dist_util.dev())
                # t = th.ones((batch.shape[0]), dtype=th.int64) * 249
                # t = t.to(dist_util.dev())
                # x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)

                # # Pendulum angle Interventions
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001
                
                
                # # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES
                
                # mu[:, :16] = th.ones((args.batch_size, 16)) * -0.32 # 80

                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                # z = reparameterize(z_post, var)


                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 4, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )

                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_angle.extend([sample.cpu().numpy() for sample in gathered_samples])



                # # Light Position INTERVENTIONS
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001
                
                # # UNIFORMLY SAMPLE A RANDOM INTERVENTION VALUE
                
                
                # # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES
                
                # mu[:, 16:32] = th.ones((args.batch_size, 16)) * -0.32 # 80

                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                # z = reparameterize(z_post, var)


                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 4, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )

                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_light.extend([sample.cpu().numpy() for sample in gathered_samples])
                
                
                # # Shadow Length INTERVENTIONS
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001
                
                # # UNIFORMLY SAMPLE A RANDOM INTERVENTION VALUE
                
                
                # # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES

                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)
                # z_post[:, 32:48] = th.ones((args.batch_size, 16)) * 1.3 # 10

                # z = reparameterize(z_post, var)



                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 4, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )

                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_shadow_len.extend([sample.cpu().numpy() for sample in gathered_samples])

                
                
                # # Shadow Position INTERVENTIONS
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001
                
                # # UNIFORMLY SAMPLE A RANDOM INTERVENTION VALUE
                
                
                # # GENERATE THE GROUND-TRUTH BASED ON CONDITIONED VALUES

                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)
                # z_post[:, 48:64] = th.ones((args.batch_size, 16)) * 1.5 # 16

                # z = reparameterize(z_post, var)


                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 4, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )

                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_shadow_pos.extend([sample.cpu().numpy() for sample in gathered_samples])
            
            elif "circuit" in args.data_dir:
                A = th.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=th.float32).to(batch.device)

                noise = th.randn_like(batch).to(dist_util.dev())
                t = th.ones((batch.shape[0]), dtype=th.int64) * 249
                t = t.to(dist_util.dev())
                
                x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)
                
                
                
                # ROBOT ARM INTERVENTIONS
                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                # mu[:, :8] = th.ones((args.batch_size, 8)) * 0.3 # 30
                mu[:, :128] = th.ones((args.batch_size, 128)) * 0.1 # 30
                
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)


                x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)


                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                cond["z"] = z
                cond["c"] = cond["c"].to(dist_util.dev())

                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    noise=x_t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cond,
                    w=w
                )

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images_arm.extend([sample.cpu().numpy() for sample in gathered_samples])
                
                

                # # BLUE LIGHT INTERVENTIONS
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001

                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)
                # # z_post[:, 8:16] = th.ones((args.batch_size, 8)) * 0.1 # 80
                # z_post[:, 128:256] = th.ones((args.batch_size, 128)) * 0.0 # 80

                # z = reparameterize(z_post, var)


                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 3, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )

                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_blue.extend([sample.cpu().numpy() for sample in gathered_samples])
                
                
                # # GREEN LIGHT INTERVENTIONS
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001

                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)
                # z_post[:, 256:384] = th.ones((args.batch_size, 128)) * 0.0 # 10

                # z = reparameterize(z_post, var)



                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 3, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )

                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_green.extend([sample.cpu().numpy() for sample in gathered_samples])

                
                
                # # RED LIGHT INTERVENTIONS
                # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                # var = th.ones(mu.shape).to(mu.device) * 0.001

                # z_pre = model.causal_mask.causal_masking(mu, A)
                # z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)
                # # z_post[:, 24:32] = th.ones((args.batch_size, 8)) * 0.9 # 16
                # z_post[:, 384:512] = th.ones((args.batch_size, 128)) * 0.0 # 16


                # z = reparameterize(z_post, var)


                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # cond["z"] = z
                # cond["c"] = cond["c"].to(dist_util.dev())

                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 3, args.image_size, args.image_size),
                #     noise=x_t,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=cond,
                #     w=w
                # )

                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images_red.extend([sample.cpu().numpy() for sample in gathered_samples])

            # if count == 5:
            break
        
        
    
        generate_interventions = True
        if generate_interventions:
            if "morphomnist" in args.data_dir:
                save_image(batch[:32], '../results/morphomnist/causaldiffae/original.png')


                # SAVE THICKNESS INTERVENED IMAGE
                arr = np.concatenate(all_images_thickness, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:32]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/morphomnist/causaldiffae_masked/intervene_thickness.png')
                save_image(temp, f'../results/morphomnist/causaldiffae/intervene_thickness_w={w}.png')

                # SAVE INTENSITY INTERVENED IMAGE
                arr = np.concatenate(all_images_intensity, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:32]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/morphomnist/causaldiffae_masked/intervene_intensity.png')
                save_image(temp, f'../results/morphomnist/causaldiffae/intervene_intensity_w={w}.png')
            elif "pendulum" in args.data_dir:
                save_image(batch[:16], '../results/pendulum/causaldiffae_masked/original.png')


                # # RECONSTRUCTION IMAGE
                # arr = np.concatenate(reconstructions, axis=0)
                # arr = arr[: args.num_samples]
                # temp = arr[:16]

                # temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/pendulum/causaldiffae_updated/reconstruction.png')
                
                
                # SAVE PENDULUM ANGLE INTERVENED IMAGE
                arr = np.concatenate(all_images_angle, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:16]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, f'../results/pendulum/causaldiffae/intervene_angle.png')
                # save_image(temp, f'../results/pendulum/causaldiffae/intervene_angle_w={w}.png')
                save_image(temp, f'../results/pendulum/causaldiffae_masked/intervene_angle_w={w}.png')



                # SAVE LIGHT POSITION INTERVENED IMAGE
                arr = np.concatenate(all_images_light, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:16]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, f'../results/pendulum/causaldiffae/intervene_light.png')
                # save_image(temp, f'../results/pendulum/causaldiffae/intervene_light_w={w}.png')
                save_image(temp, f'../results/pendulum/causaldiffae_masked/intervene_light_w={w}.png')
                
                
                # SAVE SHADOW LENGTH INTERVENED IMAGE
                arr = np.concatenate(all_images_shadow_len, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:16]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/pendulum/causaldiffae/intervene_shadowlen.png')
                # save_image(temp, f'../results/pendulum/causaldiffae/intervene_shadowlen_w={w}.png')
                save_image(temp, f'../results/pendulum/causaldiffae_masked/intervene_shadowlen_w={w}.png')
                
                
                # SAVE SHADOW POSITION INTERVENED IMAGE
                arr = np.concatenate(all_images_shadow_pos, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:16]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/pendulum/causaldiffae/intervene_shadowpos.png')
                # save_image(temp, f'../results/pendulum/causaldiffae/intervene_shadowpos_w={w}.png')
                save_image(temp, f'../results/pendulum/causaldiffae_masked/intervene_shadowpos_w={w}.png')
            
            elif "circuit" in args.data_dir:
                save_image(batch[:7], '../results/circuit/causaldiffae/original.png')


                # # RECONSTRUCTION IMAGE
                # arr = np.concatenate(reconstructions, axis=0)
                # arr = arr[: args.num_samples]
                # temp = arr[:16]

                # temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/pendulum/causaldiffae_updated/reconstruction.png')
                
                
                # SAVE PENDULUM ANGLE INTERVENED IMAGE
                arr = np.concatenate(all_images_arm, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:7]

                temp = th.tensor(temp, dtype=th.float32)
                save_image(temp, f'../results/circuit/causaldiffae/intervene_arm_w={w}.png')

                # # SAVE LIGHT POSITION INTERVENED IMAGE
                # arr = np.concatenate(all_images_blue, axis=0)
                # arr = arr[: args.num_samples]
                # temp = arr[7]

                # temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, f'../results/circuit/causaldiffae/intervene_blue_w={w}.png')

                
                # # SAVE SHADOW LENGTH INTERVENED IMAGE
                # arr = np.concatenate(all_images_green, axis=0)
                # arr = arr[: args.num_samples]
                # temp = arr[7]

                # temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, f'../results/circuit/causaldiffae/intervene_green_w={w}.png')
                
                # # # SAVE SHADOW POSITION INTERVENED IMAGE
                # arr = np.concatenate(all_images_red, axis=0)
                # arr = arr[: args.num_samples]
                # temp = arr[7]

                # temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, f'../results/circuit/causaldiffae/intervene_red_w={w}.png')
        else:
            if "morphomnist" in args.data_dir:
                mean_dist = th.tensor(sum(thickness_distances) / len(thickness_distances))
                gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
                print(f"Thickness MAE: {sum(gathered_samples) / len(gathered_samples)}")
                
                mean_dist = th.tensor(sum(intensity_distances) / len(intensity_distances))
                gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
                print(f"Intensity MAE: {sum(gathered_samples) / len(gathered_samples)}")
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
    logger.log("testing complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        rep_cond=False,
        in_channels=3,
        n_vars=2
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
