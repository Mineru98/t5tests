from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

use_weight = False
continue_train = False
model_name = "GPT-j-6B-8bit-wikipedia-finetune"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help = "model name")
args = parser.parse_args()
if args.model:
    print("=== param model name", args.model)
    model_name = args.model
else:
    print("=== model not specified")
    exit(0)

#model_name = "EleutherAI/gpt-j-6B"
#model_name = "KoboldAI/GPT-J-6B-Skein"
#model_name = "hivemind/gpt-j-6B-8bit"

tokenizer = AutoTokenizer.from_pretrained(model_name)
gpt = AutoModelForCausalLM.from_pretrained(model_name)

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