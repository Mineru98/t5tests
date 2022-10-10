# https://huggingface.co/docs/transformers/model_sharing

from huggingface_hub import notebook_login
from gpt_j_8bit import GPTJForCausalLM8, GPTJBlock8, add_adapters
from transformers import AutoTokenizer

notebook_login()

# from transformers import AutoModel, PreTrainedTokenizerFast, AutoModelWithLMHead, TFGPT2LMHeadModel, FlaxGPT2LMHeadModel

# model_path = "./gpt-j/Models/gpt-j-6B-ko-voc-to-8bit-conv"
# gpt =  GPTJForCausalLM8.from_pretrained(model_path)

# gpt.push_to_hub("gpt-j-6B-voc-ext-to-91238-8bit")

tokenizer_name = "tokenizer-gpt-j-plus-ko"
tokenizer_path = f"./train_tokenizer/{tokenizer_name}"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.push_to_hub("tokenizer-gpt-j-ext-ko")