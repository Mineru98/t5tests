# https://huggingface.co/docs/transformers/model_sharing

from huggingface_hub import notebook_login
from transformers import AutoTokenizer, logging, pipeline, AutoModelForCausalLM
from datasets import load_from_disk, load_dataset

import torch

notebook_login()

checkpoint = 2620
repo_id = f"polyglot-ko-3.8b-multi-func-{checkpoint}"

#latest_model_dir = "EleutherAI/polyglot-ko-1.3b"
latest_model_dir = f"/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func/checkpoint-{checkpoint}"
tokenizer_dir = latest_model_dir
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

gpt = AutoModelForCausalLM.from_pretrained(
    latest_model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device, torch.float16)

print("writing...")
# gpt.save_pretrained(save_directory=f"/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func-save/checkpoint-{checkpoint}", 
#                     is_main_process=False)
# tokenizer.save_pretrained(save_directory=f"/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func-save/checkpoint-{checkpoint}")

gpt.push_to_hub(repo_id=repo_id)
tokenizer.push_to_hub(repo_id=repo_id)