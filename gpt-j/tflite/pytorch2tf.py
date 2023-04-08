import torch
from transformers import OPTForCausalLM, TFOPTForCausalLM

model = TFOPTForCausalLM.from_pretrained(
    "./Models/OPT-6.7B-Erebus-instruct-native", 
    from_pt=True, 
    device_map="auto"
)
model.save_pretrained("./Models/OPT-6.7B-Erebus-instruct-native-tf")