import sys, os, argparse, transformers, torch
from datasets import load_dataset, load_metric, load_from_disk, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.contrib.concurrent import process_map
from transformers import  GPTJForCausalLM, AutoTokenizer, TrainingArguments
from transformers.optimization import AdafactorSchedule
from transformers.trainer_pt_utils import get_parameter_names
from gpt_j_8bit import GPTJForCausalLM8, GPTJBlock8, add_adapters
from bitsandbytes.optim import Adam8bit
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# gloabl vars
dataset_source = "wiki"
tokenizer_name = "tokenizer-gpt-j-plus-ko"
max_input_length = 128
continue_train = False
training_size = 0  # 0 means all
batch_size = 8     # 0 means auto
validation_data_size = batch_size * 10
base_model_type = "int8"    # fp16, int8

# accelerate
accelerator = Accelerator(
    DistributedDataParallelKwargs(find_unused_parameters=True), 
    # mixed_precision="fp16"
)
device = accelerator.device

# tokenizer
tokenizer = None

class TextDataset(Dataset):
    def tokenizing_sample(self, s):
        text = s[self.feature_name]
        tt = tokenizer(text, max_length=max_input_length, truncation=True, padding=True, return_tensors="pt")
        combined_line_list = [tt["input_ids"]]
        return combined_line_list
            
    def __init__(self, tokenize: bool = False):
        accelerator.print("reading dataset...", dataset_source)
        if dataset_source == "sns":
            ds = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/한국어 SNS/korean_sns_training_gpt2_v2.json")
            feature_name = "sample"
        elif dataset_source == "wiki":
            wiki_local = "/home/chang/nas1/linux/dataset/text/wikipedia/20221001.kr"
            if os.path.exists(wiki_local):
                ds = load_from_disk(wiki_local)
            else:
                ds = load_dataset("lcw99/wikipedia-korean-20221001")
            feature_name = "text"
        elif dataset_source == "cc100":
            ds = load_dataset("cc100", lang="ko")
            feature_name = "text"
        elif dataset_source == "namu":
            ds = load_dataset("heegyu/namuwiki-extracted")
            feature_name = "text"
        self.feature_name = feature_name
        
        ds = ds["train"]
        if training_size > 0:
            ds = ds.select(range(training_size))
        accelerator.print("reading dataset done...", dataset_source)

        if tokenize:
            examples = []
            num_worker = 12
            chunk_size = int(len(ds) / (num_worker * 4))
            if chunk_size > 100000:
                chunk_size = 100000
            accelerator.print("chunk_size=", chunk_size)
            for result in process_map(self.tokenizing_sample, ds, max_workers=num_worker, chunksize=chunk_size):
                examples += result
        else:
            examples = ds[self.feature_name]
        self.data_list = examples

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def get_dataloaders(tokenize: bool = False, loader_batch_size: int = batch_size):
    dataset = TextDataset(tokenize=tokenize)
    train_dataset = dataset[validation_data_size:]
    eval_dataset = dataset[:validation_data_size]
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=loader_batch_size, collate_fn=lambda x: x)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=loader_batch_size, collate_fn=lambda x: x)
    return train_dataloader, eval_dataloader

def build_tokenizer():
    tokenizer_path = f"../train_tokenizer/{tokenizer_name}"
    accelerator.print("\n-----------------------\ntokenizer path = ", tokenizer_path)
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        accelerator.print("downloading tokenizer from hf")
        tokenizer = AutoTokenizer.from_pretrained("lcw99/tokenizer-gpt-j-ext-ko")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_adam8bit_optimizer(model):
    training_args = TrainingArguments(per_device_train_batch_size=batch_size, output_dir=".")

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    return adam_bnb_optim
        
def init_model():
    if base_model_type == "fp16":
        max_memory_mapping = {0: "10GB", 1: "10GB"}
        load_in_8bit = True
        gpt = GPTJForCausalLM.from_pretrained(
            "./Models/gpt-j-6B-fp16-ko-voc",
            #revision="float16",
            #torch_dtype=torch.float16,
            device_map='auto',
            load_in_8bit=load_in_8bit,
            #max_memory=max_memory_mapping,
            #low_cpu_mem_usage=True,
            #use_cache=False,
        )
        # tokenizer_len = len(tokenizer)
        # print("\n\n\n=====\ntokenizer_len=", tokenizer_len)
        # gpt.resize_token_embeddings(tokenizer_len)
        # print("resize done....")
        gpt.gradient_checkpointing_enable()
        
        gpt.config.__dict__["_name_or_path"] = "lcw99/gpt-j-6B-8bit"
        # gpt.config.__dict__["use_cache"] = False
        # gpt.save_pretrained("./Models/gpt-j-6B-8bit-ko-voc")
        if True:
            #optimizer = Adam8bit(gpt.parameters(), lr=1e-5)
            optimizer = build_adam8bit_optimizer(gpt)
        else:
            optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-5)
    else:
        transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock8  
        model_path = "./Models/gpt-j-6B-ko-voc-to-8bit-conv"
        if os.path.exists(model_path):
            accelerator.print("base model path = ", model_path)
            gpt =  GPTJForCausalLM8.from_pretrained(model_path)
        else:
            hf_model = "lcw99/gpt-j-6B-voc-ext-to-91238-8bit"
            accelerator.print("downloading..", hf_model)
            gpt = GPTJForCausalLM8.from_pretrained(hf_model)
        add_adapters(gpt)
        gpt.gradient_checkpointing_enable()
        gpt.config.__dict__["_name_or_path"] = "lcw99/gpt-j-6B-8bit-custom"

        optimizer = Adam8bit(gpt.parameters(), lr=1e-5, weight_decay=0.01)
        
    gpt.config.__dict__["use_cache"] = False

    return gpt, optimizer
              
def _get_lr(param_group, param_state):
    step = param_state["step"]
    eps = param_group["eps"]
    return 5e-5 - eps * step * 1e-2
              
def loss_function(output, input):
    loss_cross_entropy = nn.CrossEntropyLoss()
    loss = loss_cross_entropy(output.logits[:, :-1, :].flatten(0, -2), input['input_ids'][:, 1:].flatten())  # not woring on int
    return loss
    # else:
    # if output.loss:
    #     return output.loss
    loss = F.cross_entropy(output.logits[:, :-1, :].flatten(0, -2), input['input_ids'][:, 1:].flatten(), reduction='mean')
    return loss
    
def trainer():
    global batch_size
    if batch_size == 0:
        batch_size = 4
    model, optimizer = init_model()
    train_dataloader, eval_dataloader = get_dataloaders(tokenize=False, loader_batch_size=batch_size)
 
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(num_training_steps*0.1), num_training_steps
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        batch_token = tokenizer(batch, truncation=True, padding=True, max_length=max_input_length, return_tensors='pt')
        with torch.cuda.amp.autocast():
            outputs = model(**batch_token.to(device))
            loss = loss_function(outputs, batch_token)
        # accelerator.backward(loss)
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
                
        lr_scheduler.step()
        if i % 10 == 0 and i != 0:
            state = {
                'k' : i, 'epoch': num_epochs, 
                'lr_scheduler': lr_scheduler.state_dict(), 
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }   
            print(loss)
            # torch.save(state, "./TestModel/model.pt")
                                    
def main():
    global dataset_source, tokenizer_name, max_input_length, continue_train, training_size, batch_size, tokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--continue_train", action='store_true', help = "continue trainning")
    parser.add_argument("-d", "--dataset", help = "dataset source = [sns, wiki, cc100, namu]")
    parser.add_argument("-t", "--tokenizer", help = "tokenizer name")
    parser.add_argument("-i", "--max_input_length", help = "max input length")
    parser.add_argument("-s", "--training_size", help = "training size, 0 for all")
    parser.add_argument("-b", "--batch_size", help = "batch size, 0 for auto")

    args = parser.parse_args()

    if args.dataset:
        dataset_source = args.dataset
    else:
        parser.print_help(sys.stderr)
        return

    if args.tokenizer:
        tokenizer_name = args.tokenizer

    if args.continue_train:
        print("=== param continue trainning")
        continue_train = True

    if args.max_input_length:
        max_input_length = int(args.max_input_length)
    if args.training_size:
        training_size = int(args.training_size)
    if args.batch_size:
        batch_size = int(args.batch_size)
        
    tokenizer = build_tokenizer()
    
    trainer()
    
if __name__ == '__main__':
    sys.exit(main()) 