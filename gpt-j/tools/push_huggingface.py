# https://huggingface.co/docs/transformers/model_sharing

from huggingface_hub import notebook_login
from transformers import AutoTokenizer, logging, pipeline, AutoModelForCausalLM
from datasets import load_from_disk, load_dataset

import torch

notebook_login()

repo_id = f"polyglot-ko-12.8b-chang-instruct-chat"

#latest_model_dir = "EleutherAI/polyglot-ko-1.3b"
latest_model_dir = f"/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-12.8b-sharegpt-step4-finalize/ckpt-120-on-test-deploy"
tokenizer_dir = latest_model_dir
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("loading...")

gpt = AutoModelForCausalLM.from_pretrained(
    latest_model_dir,
    torch_dtype=torch.float16,
)

print("writing...")
# gpt.save_pretrained(save_directory=f"/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func-save/checkpoint-{checkpoint}", 
#                     is_main_process=False)
# tokenizer.save_pretrained(save_directory=f"/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func-save/checkpoint-{checkpoint}")

gpt.push_to_hub(repo_id=repo_id)
tokenizer.push_to_hub(repo_id=repo_id)