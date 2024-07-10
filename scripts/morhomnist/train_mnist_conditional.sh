mpiexec -n 6 python image_train.py --data_dir ../datasets/morphomnist_data/ --image_size 28 --num_channels 128 --num_res_blocks 3 --learn_sigma False --class_cond True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 128