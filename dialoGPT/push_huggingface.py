# https://huggingface.co/docs/transformers/model_sharing

from huggingface_hub import notebook_login

notebook_login()

from transformers import AutoModel, PreTrainedTokenizerFast, AutoModelWithLMHead, TFGPT2LMHeadModel, FlaxGPT2LMHeadModel

model_name = "dialoGPT-base-korean-chit-chat-v2"
hf_model_name = "dialoGPT-medium-korean-chit-chat"
model_dir = f"./Models/{model_name}/save-checkpoint-180000"

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
pt_model = AutoModelWithLMHead.from_pretrained(model_dir)
tf_model = TFGPT2LMHeadModel.from_pretrained(model_dir, from_pt=True)
flax_model = FlaxGPT2LMHeadModel.from_pretrained(model_dir, from_pt=True)

tokenizer.push_to_hub(hf_model_name)
pt_model.push_to_hub(hf_model_name)
tf_model.push_to_hub(hf_model_name)
flax_model.push_to_hub(hf_model_name)