# https://huggingface.co/docs/transformers/model_sharing

from huggingface_hub import notebook_login

notebook_login()

from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration, FlaxT5ForConditionalGeneration
from transformers import AutoTokenizer, T5TokenizerFast

model_name = "mt5-base-korean-text-summary"
model_dir = f"./Models/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
pt_model = T5ForConditionalGeneration.from_pretrained(model_dir)
tf_model = TFT5ForConditionalGeneration.from_pretrained(model_dir, from_pt=True)
flax_model = FlaxT5ForConditionalGeneration.from_pretrained(model_dir, from_pt=True)

tokenizer.push_to_hub(model_name)
pt_model.push_to_hub(model_name)
tf_model.push_to_hub(model_name)
flax_model.push_to_hub(model_name)