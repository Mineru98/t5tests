accelerate launch --config_file deepspeed_stage3.yml train_gpt_j.py --config_file configs/polyglot/12.8B/polyglot-ko-12.8B-2gpu-oddqkv-saju-hf.json
accelerate launch --config_file deepspeed_stage3.yml train_gpt_j.py --config_file configs/polyglot/12.8B/polyglot-ko-12.8B-2gpu-oddqkv-saju-instruct.json
accelerate launch --config_file deepspeed_stage3.yml train_gpt_j.py --config_file configs/polyglot/12.8B/polyglot-ko-12.8B-2gpu-oddqkv-saju-finalize.json
