from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")

gpt = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-5.8b", torch_dtype=torch.float)
#gpt.config.__dict__["_name_or_path"] = f"lcw99/gpt-neo-1.3B-ko"

gpt.save_pretrained("./StockModels/polyglot-ko-5.8b-fp32")
tokenizer.save_pretrained("./StockModels/polyglot-ko-5.8b-fp32")

# gpt.push_to_hub("gpt-neo-1.3B-ko-fp16")
# tokenizer.push_to_hub("gpt-neo-1.3B-ko-fp16")