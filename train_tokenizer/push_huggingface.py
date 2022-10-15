# https://huggingface.co/docs/transformers/model_sharing

from huggingface_hub import notebook_login
from transformers import AutoTokenizer
from transformers import AutoModel, PreTrainedTokenizerFast, AutoModelWithLMHead, TFGPT2LMHeadModel, FlaxGPT2LMHeadModel
from datasets import load_from_disk

notebook_login()

tokenizer_name = "tokenizer-wiki-plus-namu-kogpt2-base-gpt-j"
tokenizer_path = f"./{tokenizer_name}"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.push_to_hub(tokenizer_name)
    