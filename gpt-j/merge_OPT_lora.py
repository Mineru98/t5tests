import os

import torch
import transformers
from peft import PeftModel, PeftConfig
from transformers import OPTForCausalLM, AutoTokenizer  # noqa: F402

LORA_ADAPTER = "/home/chang/AI/llm/t5tests/gpt-j/Models/OPT-6.7B-Erebus-instruct/final"

config = PeftConfig.from_pretrained(LORA_ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

base_model = OPTForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.decoder.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.decoder.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights
for layer in lora_model.base_model.model.model.decoder.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

OPTForCausalLM.save_pretrained(
    base_model, "./Models/OPT-6.7B-Erebus-instruct-hf", state_dict=deloreanized_sd, max_shard_size="5000MB"
)