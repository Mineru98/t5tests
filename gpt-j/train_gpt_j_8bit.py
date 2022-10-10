
"""## Load the dataset"""

import transformers
from datasets import load_dataset, load_metric, load_from_disk, Dataset
import evaluate
import random, pickle, os, glob

import nltk
nltk.download('punkt')
import string
from transformers import  PreTrainedTokenizerFast, AutoConfig, PretrainedConfig, AutoTokenizer

import argparse
import sys

continue_train = False
token_expand = True
max_input_length = 256

batch_size = 0
trainning_size = 0
val_data_size = 32
model_size = "medium" # small, medium
dataset_source = "wiki" # sns, wiki
#feature_name = "text" # sample, text

num_train_epochs = 10
huggingface_trainner = True

model_name_base = "GPT-j-6B-8bit"
tokenizer_name = "tokenizer-gpt-j-plus-ko"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--continue_train", help = "continue trainning")
parser.add_argument("-d", "--dataset", help = "dataset source = [sns, wiki, cc100]")
parser.add_argument("-t", "--tokenizer", help = "tokenizer name")
parser.add_argument("-i", "--max_input_length", help = "max input length")
parser.add_argument("-s", "--trainning_size", help = "trainning size, 0 for all")
parser.add_argument("-b", "--batch_size", help = "batch size, 0 for auto")

args = parser.parse_args()

if args.dataset:
    dataset_source = args.dataset
else:
    parser.print_help(sys.stderr)
    exit(0)

if args.tokenizer:
    tokenizer_name = args.tokenizer

if args.continue_train:
    print("=== param continue trainning")
    continue_train = True

if args.max_input_length:
    max_input_length = int(args.max_input_length)
if args.trainning_size:
    trainning_size = int(args.trainning_size)
if args.batch_size:
    batch_size = int(args.batch_size)
    
model_name = f'{model_name_base}_{tokenizer_name}_{dataset_source}_{max_input_length}' 
dataset_cache_path = f"./cache/{model_name}_{trainning_size}"
tokenizer_path = f"../train_tokenizer/{tokenizer_name}"

print("--------------------")
print("trainning:", model_name)
print("--------------------")

model_checkpoint = "hivemind/gpt-j-6B-8bit"

model_dir = f"./Models/{model_name}"



# print(medium_datasets)

# """## Dataset train/validation/test split"""

# datasets_train_test = medium_datasets["train"].train_test_split(test_size=5000)
# datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=5000)

# medium_datasets["train"] = datasets_train_validation["train"]
# medium_datasets["validation"] = datasets_train_validation["test"]
# medium_datasets["test"] = datasets_train_test["test"]

# print(medium_datasets)

# n_samples_train = len(medium_datasets["train"])
# n_samples_validation = len(medium_datasets["validation"])
# n_samples_test = len(medium_datasets["test"])
# n_samples_total = n_samples_train + n_samples_validation + n_samples_test

# print(f"- Training set: {n_samples_train*100/n_samples_total:.2f}%")
# print(f"- Validation set: {n_samples_validation*100/n_samples_total:.2f}%")
# print(f"- Test set: {n_samples_test*100/n_samples_total:.2f}%")

# # keep only a subsample of the datasets
# if step_factor == 1:
#     medium_datasets["train"] = medium_datasets["train"].select(range(11000))
# else:
#     if dataset_source == "sns":
#         medium_datasets["train"] = medium_datasets["train"].select(range(1000000))
#     else:
#         medium_datasets["train"] = medium_datasets["train"]
# medium_datasets["validation"] = medium_datasets["validation"].select(range(5 * step_factor))
# medium_datasets["test"] = medium_datasets["test"].select(range(5 * step_factor))

# print(medium_datasets)

# """## Data preprocessing"""


print("\n-----------------------\ntokenizer_path = ", tokenizer_path)
if os.path.exists(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
else:
    print("downloading tokenizer from hf")
    tokenizer = AutoTokenizer.from_pretrained("lcw99/tokenizer-gpt-j-ext-ko")

tokenizer.pad_token = tokenizer.eos_token

def combine_lines(ids):
    combined_line = []
    combined_line_list = []
    for l in ids:
        l = l[:max_input_length-1]
        if len(combined_line) + len(l) < max_input_length:
            combined_line += [tokenizer.eos_token_id] + l
        else:
            combined_line_list.append(combined_line)
            combined_line = l
    if len(combined_line) > 0:
        combined_line_list.append(combined_line)        
    #str = tokenizer.batch_decode(combined_line_list, skip_special_tokens=False)
    return combined_line_list

def build_list_from_dataset(ds, feature_name):
    examples = []
    for s in ds:
        lines = s[feature_name].split("\n")
        lines = list(filter(lambda l: len(l) > 0, lines))
        if len(lines) == 0:
            continue
        tt = tokenizer(lines, max_length=max_input_length, truncation=True, padding=False)
        combined_line_list = combine_lines(tt["input_ids"])
        examples += combined_line_list
    examples = combine_lines(examples)
    return examples
        
def get_dataset():
    print("reading dataset...", dataset_source)
    if dataset_source == "sns":
        ds = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/한국어 SNS/korean_sns_training_gpt2_v2.json")
        feature_name = "sample"
    elif dataset_source == "wiki":
        ds = load_from_disk("/home/chang/nas1/linux/dataset/text/wikipedia/20221001.kr")
        feature_name = "text"
    elif dataset_source == "cc100":
        # cc100_local = "./dataset/cc100.kr"
        # if os.path.exists(cc100_local):
        #     ds = load_from_disk(cc100_local)    
        # else:
        #     ds = load_dataset("cc100", lang="ko")
        #     ds.save_to_disk(cc100_local)
        ds = load_dataset("cc100", lang="ko")
        feature_name = "text"
    ds = ds["train"]
    print("reading dataset done...", dataset_source)
    return ds, feature_name
    
def get_data_list(eval_size: int = 100):
    ds, feature_name = get_dataset()
    if trainning_size > 0:
        ds = ds.select(range(trainning_size))
    return {
        "input_ids": build_list_from_dataset(ds, feature_name),
    }

if os.path.exists(dataset_cache_path):
    print("loading form cashed dataset...")
    datasets = load_from_disk(dataset_cache_path)    
else:
    print("start building dataset...")
    dict = get_data_list(val_data_size)
    datasets = Dataset.from_dict(dict)
    datasets = datasets.train_test_split(test_size = val_data_size)
    print("\n------------------\n")
    print(datasets)
    datasets.save_to_disk(dataset_cache_path)

# def preprocess_data(examples):
#     print("len(examples)=", len(examples[f"{feature_name}"]))        
#     samples = []
#     for i in range(len(examples[f"{feature_name}"])):
#         str = examples[f"{feature_name}"][i]
#         sample_split = str.splitlines()
#         for s in sample_split:
#             #print(s)
#             samples.append(s)
#     inputs = samples
#     #print(samples)
#     print("len(samples)=", len(samples))        
#     #exit(0)
#     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=False)
#     return model_inputs

# medium_datasets_cleaned = medium_datasets
# train_data_len = len(medium_datasets_cleaned["train"])
# print("no_train_data(filterd)=", train_data_len)

# if huggingface_trainner:
#     if step_factor == 1:
#         dataset_cache_path += "_small"
#     if os.path.exists(dataset_cache_path):
#         tokenized_datasets = load_from_disk(dataset_cache_path)    
#     else:
#         tokenized_datasets = medium_datasets_cleaned.map(preprocess_data, batched=True, load_from_cache_file=True)
#         print("done loading:", tokenized_datasets)
#         tokenized_datasets.save_to_disk(dataset_cache_path)

# print("\n\n==========================\ntokenized_datasets")
# print(tokenized_datasets)

from transformers import AutoModelWithLMHead, TrainingArguments, \
                          DataCollatorForLanguageModeling, Trainer, GPTJConfig



data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="pt", mlm=False)

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#rouge = load_metric("rouge")
metric_accuracy = evaluate.load("accuracy")
#perplexity = evaluate.load("perplexity", module_type="metric")



from transformers import GPTJForCausalLM
from transformers.optimization import Adafactor, AdafactorSchedule
from gpt_j_8bit import GPTJForCausalLM8, GPTJBlock8, add_adapters
import time, os
import torch.nn.functional as F
from bitsandbytes.optim import Adam8bit

if False:
    max_memory_mapping = {0: "10GB", 1: "10GB"}
    gpt = GPTJForCausalLM.from_pretrained(
        #"EleutherAI/gpt-j-6B",
        "./Models/gpt-j-6B-fp16-ko-voc",
        revision="float16",
        torch_dtype=torch.float16,
        #low_cpu_mem_usage=True,
        use_cache=False,
        device_map='auto',
        load_in_8bit=True,
        #gradient_checkpointing=True,
        #max_memory=max_memory_mapping,
    )
    # tokenizer_len = len(tokenizer)
    # print("\n\n\n=====\ntokenizer_len=", tokenizer_len)
    # gpt.resize_token_embeddings(tokenizer_len)
    # print("resize done....")
    gpt.config.__dict__["_name_or_path"] = "lcw99/gpt-j-6B-8bit"
    gpt.config.__dict__["use_cache"] = False
    gpt.save_pretrained("./Models/gpt-j-6B-8bit-ko-voc")
    print("save done....")
else:
    transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock8  
    model_path = "./Models/gpt-j-6B-ko-voc-to-8bit-conv"
    if os.path.exists(model_path):
        print("base model path = ", model_path)
        gpt =  GPTJForCausalLM8.from_pretrained(model_path)
    else:
        hf_model = "lcw99/gpt-j-6B-voc-ext-to-91238-8bit"
        print("downloading..", hf_model)
        gpt = GPTJForCausalLM8.from_pretrained(hf_model)
    add_adapters(gpt)
    gpt.gradient_checkpointing_enable()

gpt.config.__dict__["_name_or_path"] = "lcw99/gpt-j-6B-8bit"
gpt.config.__dict__["use_cache"] = False
gpt.to('cuda')
     
# class MyTrainer(Trainer):    
#     def create_optimizer_and_scheduler(self, num_training_steps):
#         self.optimizer = Adam8bit(self.model.parameters(), lr=1e-5)
#         self.lr_scheduler = AdafactorSchedule(optimizer)    

#     def unwrap_model(self, model: nn.Module) -> nn.Module:
#         if hasattr(model, "module"):
#             return self.unwrap_model(model.module)
#         else:
#             return model

#     def compute_loss(self, model, inputs, return_outputs=False):
#         outputs = model(**inputs)
#         # Save past state if it exists
#         # TODO: this needs to be fixed and made cleaner later.
#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]

#         loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), inputs['input_ids'][:, 1:].flatten(),
#                                reduction='mean')

#         #print("loss=", loss)
#         return (loss, outputs) if return_outputs else loss


from tqdm.auto import tqdm

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    l0 = logits[0]
    pred_ids = torch.argmax(l0, dim=-1)
    #pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    return pred_ids, labels

def compute_metrics(eval_pred):
    # preds, labels = eval_pred
    # # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # # by preprocess_logits_for_metrics
    # labels = labels.reshape(-1)
    # preds = preds[0].reshape(-1)
    # mask = labels != -100
    # labels = labels[mask]
    # preds = preds[mask]
    # acc = metric_accuracy.compute(predictions=preds, references=labels)
    # return acc
        
    labels_ids = eval_pred.label_ids
    pred_ids = eval_pred.predictions[0]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    ppl = {}
    ppl["mean_perplexity"] = 0.0

    #ppl = perplexity.compute(predictions=pred_str_filterd, model_id='gpt2')

    print("\n===========predictions\n", pred_str)
    print()
    for label in random.sample(list(label_str), 2):
        print("label=", label)

    return {
        "mean_perplexity": round(ppl["mean_perplexity"], 4)
    }

import math
def _get_lr(param_group, param_state):
    step = param_state["step"]
    eps = param_group["eps"]
    return 5e-5 - eps * step * 1e-3

if huggingface_trainner:
    if batch_size == 0:
        auto_find_batch_size = True
        batch_size = 8
    else:
        auto_find_batch_size = False
    args = TrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        auto_find_batch_size=auto_find_batch_size,
        gradient_accumulation_steps=1,
        weight_decay=0.02,
        save_total_limit=5,
        num_train_epochs=num_train_epochs,
        #predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        ignore_data_skip=True,     # set true for ignore batch skip, fast
    )
    
    optimizer = Adam8bit(gpt.parameters(), lr=1e-5, min_8bit_size=16384)
    lr_scheduler = AdafactorSchedule(optimizer)    
    optimizer._get_lr = _get_lr
    trainer = Trainer(
        model=gpt,
        args=args,
        train_dataset = datasets["train"],
        eval_dataset = datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(optimizer, lr_scheduler)
    )

    print("start trainning -----------------------------")
    if continue_train:
        trainer.train(True)
    else:
        trainer.train()
    trainer.save_model()

else:

    dataset, feature_name = get_dataset()
    
    #model =  GPTJForCausalLM8.from_pretrained(model_checkpoint, low_cpu_mem_usage=True)
    #configuration = GPTJConfig()

    #gpt = GPTJModel(model.config)
    #gpt = model
    #add_adapters(gpt)
    #gpt.to('cuda')
    gpt.gradient_checkpointing_enable()

    # example dataset
    #dataset = load_dataset("transformersbook/codeparrot-train", streaming=True)

    optimizer = Adam8bit(gpt.parameters(), lr=1e-5)

    # Set the model to training mode
    start = time.time()

    # Training loop
    with torch.cuda.amp.autocast():
        for row in tqdm(dataset):
            if len(row[feature_name]) <= 1:
                continue
            batch = tokenizer(row[feature_name], truncation=True, max_length=128, return_tensors='pt')
            batch = {k: v.cuda() for k, v in batch.items()}
            out = gpt.forward(**batch,)
            loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                                reduction='mean')
            print(loss)
            loss.backward()
            optimizer.step()

    print("Finished fine-tuning in {}".format(time.time() - start))

    # --------------> Saving fine-tuned model <-----------------#
    try:
        save_dir = "./finetuned_gpt-j-8_bit"
        os.makedirs(save_dir)
        gpt.save_pretrained(save_dir)
    except Exception as e:
        #print("Error saving model: ", e)
        print("Error saving model: {}".format(e))