from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
from peft import PeftConfig, PeftModel, get_peft_model

tokenizer = AutoTokenizer.from_pretrained("/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-5.8b-lora/checkpoint-60")

peft_config = PeftConfig.from_pretrained("/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-5.8b-lora/checkpoint-60")
gpt = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
        return_dict=True,
        torch_dtype=torch.float16)
#gpt.config.__dict__["_name_or_path"] = f"lcw99/gpt-neo-1.3B-ko"
peft_config["target_modules"] = ["query_key_value"] 
gpt = get_peft_model(gpt, peft_config)

gpt.save_pretrained("/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-5.8b-lora/checkpoint-60/lora")
tokenizer.save_pretrained("/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-5.8b-lora/checkpoint-60/lora")

# gpt.push_to_hub("gpt-neo-1.3B-ko-fp16")
# tokenizer.push_to_hub("gpt-neo-1.3B-ko-fp16")