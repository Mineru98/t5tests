from transformers import AutoTokenizer, logging, pipeline, AutoModelForCausalLM
import torch, deepspeed
from deepspeed.module_inject.containers.gptneox import DS_GPTNEOXContainer, GPTNEOXLayerPolicy
from transformers import GPTNeoXLayer

latest_model_dir_on_test = "/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func/checkpoint-2840"
gpt_on_test = AutoModelForCausalLM.from_pretrained(
    latest_model_dir_on_test,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    #load_in_8bit=True,
    #device_map='auto',
).to('cuda:0', torch.float16)

ds_engine = deepspeed.init_inference(
    gpt_on_test,
    mp_size=1,
    dtype=torch.float16,
    replace_method='auto',
    checkpoint=None,
    replace_with_kernel_inject=False,
    injection_policy={GPTNeoXLayer: (GPTNEOXLayerPolicy, )}
)
    