mpiexec -n 5 python image_conditional_test.py --data_dir ../datasets/pendulum/ --model_path ../results/pendulum/label_conditional/model_checkpoint.pt --in_channels 4 --image_size 96 --num_channels 128 --num_res_blocks 3 --learn_sigma False --context_cond True --diffusion_steps 1000 --noise_schedule linear --batch_size 16 --timestep_respacing 250 --use_ddim True