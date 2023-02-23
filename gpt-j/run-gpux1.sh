accelerate launch --config_file deepspeed_1gpu.yml train_gpt_j.py --config_file configs/polyglot-ko-5.8B-fine-tune-1gpu-wikiqna.json
accelerate launch --config_file deepspeed_1gpu.yml --main_process_port 20688 train_gpt_j.py --config_file configs/polyglot-ko-3.8B-fine-tune-1gpu-multi-func-peft.json
