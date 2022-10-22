accelerate launch --config_file deepspeed.yml --num_cpu_threads_per_process 5 train_gpt_j.py --config_file configs/gpt-neo-125M.json

accelerate launch --config_file deepspeed.yml --num_cpu_threads_per_process 3 train_gpt_j.py --config_file configs/gpt-neo-1.3B.json