from transformers import  GPTJForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainingArguments, \
                            DataCollatorForLanguageModeling, pipeline, GPTNeoForCausalLM, AutoConfig, GPTNeoModel
                            
model_file = "./StockModels/gpt-neo-300M/config.json"
gpt_config = AutoConfig.from_pretrained(model_file)
target = GPTNeoForCausalLM(gpt_config)

source_model = "./Models/gpt-neo-125M-512/checkpoint-9240"
source = GPTNeoForCausalLM.from_pretrained(source_model)

params_source = dict(source.named_parameters());
params_target = dict(target.named_parameters());

for key in params_source.keys():
    print(key, params_target[key].data.shape, params_source[key].data.shape) 
    #params_target[key].data.copy_(params_source[key].data)

#target.save_pretrained("./Models/gpt-neo-300M-transfered-from-125M")