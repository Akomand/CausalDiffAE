mpiexec -n 5 python image_causaldae_test.py --data_dir ../datasets/pendulum/ --model_path ../results/pendulum/causaldiffae_masked/model035000.pt --n_vars 4 --in_channels 4 --image_size 96 --num_channels 128 --num_res_blocks 3 --learn_sigma False --class_cond False --causal_modeling True --rep_cond True --diffusion_steps 1000 --batch_size 16 --timestep_respacing 250 --use_ddim True