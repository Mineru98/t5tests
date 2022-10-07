import os, glob
import transformers
import torch
from gpt_j_8bit import GPTJBlock, GPTJForCausalLM, GPTJModel, add_adapters, Adam8bit
from transformers import AutoTokenizer, logging, pipeline
import argparse

transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

model_name = "GPT-j-6B-8bit-wikipedia-finetune"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help = "model name")
parser.add_argument("-l", "--local_model", help = "local model name")
args = parser.parse_args()
if args.local_model:
    print("=== param using local model", args.local_model)
    model_name = args.local_model
    model_dir = f"./Models/{model_name}"
    latest_model_dir = max(glob.glob(os.path.join(model_dir, 'checkpoint-*/')), key=os.path.getmtime)
    tokenizer_dir = latest_model_dir
if args.model:
    print("=== param model name", args.model)
    model_name = args.model
    latest_model_dir = model_name
    if model_name == "hivemind/gpt-j-6B-8bit":
        tokenizer_dir = "EleutherAI/gpt-j-6B"
    else:
        tokenizer_dir = latest_model_dir

print("\n---------------------------")
print("model dir=", latest_model_dir)
print("---------------------------\n")

logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
gpt = GPTJForCausalLM.from_pretrained(latest_model_dir).to(device)
tokenizer.pad_token = tokenizer.eos_token

text_generation = pipeline(
    "text-generation",
    model=gpt,
    tokenizer=tokenizer,
    device=0
)

while True:
    text = input("Input: ")
    generated = text_generation(
        text,
        max_length=300,
        do_sample=True,
        num_return_sequences=5,
        top_p=0.95,
        top_k=50
    )

    print(*generated, sep="\n")