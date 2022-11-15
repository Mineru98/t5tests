from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, AutoConfig, GPTNeoForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./StockModels/gpt-neo-1.3B-ko-2860")

gpt = GPTNeoForCausalLM.from_pretrained("./StockModels/gpt-neo-1.3B-ko-2860", torch_dtype=torch.float16)
#gpt.config.__dict__["_name_or_path"] = f"lcw99/gpt-neo-1.3B-ko"

#gpt.save_pretrained("./StockModels/gpt-neo-1.3B-ko-fp16")
#tokenizer.save_pretrained("./StockModels/gpt-neo-1.3B-ko-fp16")

gpt.push_to_hub("gpt-neo-1.3B-ko-fp16")
tokenizer.push_to_hub("gpt-neo-1.3B-ko-fp16")