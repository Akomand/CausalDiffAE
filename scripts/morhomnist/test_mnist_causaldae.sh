python image_causaldae_test.py --data_dir ../datasets/morphomnist_data/ --model_path ../results/morphomnist/causaldiffae/model014000.pt --n_vars 2 --in_channels 1 --image_size 28 --num_channels 128 --num_res_blocks 3 --learn_sigma False --class_cond True --causal_modeling True --rep_cond True --diffusion_steps 1000 --batch_size 16 --timestep_respacing 250 --use_ddim True