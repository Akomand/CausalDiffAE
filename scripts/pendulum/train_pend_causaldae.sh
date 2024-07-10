mpiexec -n 5 python image_train.py --data_dir ../datasets/pendulum/ --in_channels 4 --image_size 96 --num_channels 128 --causal_modeling True --num_res_blocks 3 --learn_sigma False --class_cond False --rep_cond True --flow_based False --masking True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 32