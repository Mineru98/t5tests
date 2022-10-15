# https://huggingface.co/docs/transformers/model_sharing

from huggingface_hub import notebook_login
from transformers import AutoTokenizer
from transformers import AutoModel, PreTrainedTokenizerFast, AutoModelWithLMHead, TFGPT2LMHeadModel, FlaxGPT2LMHeadModel
from datasets import load_from_disk

notebook_login()

tokenizer_name = "tokenizer_wiki_plus_namu_gpt_j"
tokenizer_path = f"./train_tokenizer/{tokenizer_name}"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.push_to_hub(tokenizer_name)
    
