import torch
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Sharding Hugging Face models')
parser.add_argument('--sharding_factor', default=4, type=int, help='Sharding factor - aka how many shards to create')
parser.add_argument('--source_model_path', default="t5-v1_1-xl", type=str, help='Relative path to the source model folder')
parser.add_argument('--sharded_model_path', default="t5-v1_1-xl-sharded", type=str, help='Relative path to the target sharded model folder')
args = parser.parse_args()

def get_memory_footprint_param(param):
    r"""
        Get the memory footprint of a parameter.
    """
    return sum([param.nelement()*param.element_size()])


def get_index_json_file():
    r"""
        Get the default index.json dictionary. This
        had to contain the metadata of the model as well
        as the weight_map.
    """
    index_dict = {
        "metadata": {
            "total_size": 0
        },
        "weight_map": {}
    }
    return index_dict


def save_index_file(path_sharded, index_dict):
    r"""
        Save the index.json file.
    """
    with open(os.path.join(path_sharded, "pytorch_model.bin.index.json"), "w", encoding="utf-8") as f:
        json_config = json.dumps(index_dict, indent=2, sort_keys=True) + "\n"
        f.write(json_config)

if __name__ == "__main__":
    # Get the args
    ROOT_PATH=os.getcwd()
    source_model = args.source_model_path
    target_model = args.sharded_model_path

    path_model = os.path.join(ROOT_PATH, source_model, "pytorch_model.bin")
    path_sharded = os.path.join(ROOT_PATH, target_model)
    sharding_factor = args.sharding_factor

    # Initialize the variables
    index_dict = get_index_json_file()
    state_dict = torch.load(path_model)
    
    sharded_state_dict = {}
    total_keys = []

    current_file_name = f"pytorch_model_00001-of-{str(sharding_factor).zfill(5)}.bin"
    checking_step = len(state_dict.keys())//sharding_factor

    # Loop over the parms and shard them if necessary
    for i, key in enumerate(state_dict.keys()):
        # Get the current param
        param = state_dict[key]
        index_dict["metadata"]["total_size"] += get_memory_footprint_param(param)
        index_dict["weight_map"][key] = current_file_name
        sharded_state_dict[key] = param
        total_keys.append(key)
        # Check if we need to create a new file
        if (i+1) % checking_step == 0:
            torch.save(sharded_state_dict, os.path.join(path_sharded, current_file_name))
            sharded_state_dict = {}
            new_index = ((i+1)//checking_step) + 1
            current_file_name = f"pytorch_model_{str(new_index).zfill(5)}-of-{str(sharding_factor).zfill(5)}.bin"

    # Save the last sharded file if necessary
    if len(sharded_state_dict) > 0:
        torch.save(sharded_state_dict, os.path.join(path_sharded, current_file_name))
    
    # Last sanity check
    if total_keys != list(state_dict.keys()):
        raise ValueError("The keys in the index.json file are not the same as the keys in the model")

    save_index_file(path_sharded, index_dict)