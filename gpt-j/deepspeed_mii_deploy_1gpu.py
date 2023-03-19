import mii
mii_configs = {"tensor_parallel": 1, "dtype": "fp16", "port_number": 50950,}
model_hidden_size=3072
ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "aio": {
        "block_size": 262144,
        "queue_depth": 32,
        "thread_count": 1,
        "single_submit": True,
        "overlap_events": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "none",
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.1 * model_hidden_size * model_hidden_size,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "train_micro_batch_size_per_gpu": 1,
}

ds_config2 = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "none",
        },
    },
    "train_micro_batch_size_per_gpu": 1,
}

model_path = "/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func-v2/checkpoint-4620"

mii.deploy(task="text-generation",
           model=model_path,
           #model="EleutherAI/gpt-neox-20b",
           model_path=model_path,
           deployment_name="lcw_deployment",
           enable_deepspeed=False,
           enable_zero=True,
           ds_config=ds_config,
           mii_config=mii_configs)

