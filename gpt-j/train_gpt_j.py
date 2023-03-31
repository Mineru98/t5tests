from multiprocessing import Pool
import sys, os, argparse, transformers, torch, random, evaluate, numpy, re, json, ftfy, glob
from datasets import load_dataset, load_metric, load_from_disk, Dataset, concatenate_datasets, disable_caching
from accelerate import Accelerator, DistributedDataParallelKwargs
import accelerate
from tqdm.contrib.concurrent import process_map
from transformers import  AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainingArguments, \
                            DataCollatorForLanguageModeling, pipeline, GPTNeoForCausalLM, AutoConfig, GPTNeoModel, default_data_collator
from transformers.optimization import AdafactorSchedule
from transformers.trainer_pt_utils import get_parameter_names, nested_detach, IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
from bitsandbytes.optim import Adam8bit
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from gpt_j_8bit import GPTJForCausalLM8, GPTJBlock8, add_adapters
import pandas

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PrefixTuningConfig

data_build_only = False
gpt_neo = None      
model_file = None
save_path = None
ignore_data_skip = True
deepspeed_config_json = None
start_model_path = None

scratch = False
kor_voca_extention = False
eval_sample = False
skip_eval = False
unfreeze = []    

num_train_epochs = 2
dataset_source = ["wiki"]
max_input_length = 128
continue_train = False
training_size = 0  # 0 means all
batch_size = 8    # 0 means auto
validation_data_size = batch_size * 10
train_dataset_size = 0
optimizer_8bit = False
reset_weight = False
LoRa = False
PrefixTuning = False
softembeddings = False

model_name = None 
model_save_dir = None
train_dataloader = None 
eval_dataloader = None
# accelerate
accelerator = Accelerator(
    DistributedDataParallelKwargs(find_unused_parameters=True), 
    # mixed_precision="fp16"
)
device = accelerator.device

# tokenizer
tokenizer = None

last_eval_model = None
base_model_name = None

cache_folder_name = "Cache2"
new_model_name = "lcw99/no-name"

def name_to_filename(name):
    return name.replace("/", "_").replace(".", "_")

def tokenize_string(s):
    tt = tokenizer(f"{s}\n{tokenizer.eos_token}{tokenizer.eos_token}")
    encode_len = len(tt['input_ids']) 
    return encode_len, tt['input_ids'], tt['attention_mask']
    
def preprocess_function(ss):
    max_length = 128

    batch_size = len(ss['passage'])

    tt = len(text_templates) - 1
    inputs = [f"{s} 요약하시오.\n" for s in ss['passage']]
    targets = [f"{s}" for s in ss['summary1']]
    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length)
    labels = tokenizer(targets, padding='max_length', truncation=True, max_length=max_length)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        #print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids 
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    #print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id]*(max_length-len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"][i] = [0]*(max_length-len(sample_input_ids)) + model_inputs["attention_mask"][i]
        labels["input_ids"][i] =  [-100]*(max_length-len(sample_input_ids)) + label_input_ids 
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length]) 
    model_inputs["labels"] = labels["input_ids"]
    model_inputs.pop('token_type_ids', None)
    return model_inputs

    
def tokenizing_sample(ss):
    if softembeddings or PrefixTuning:
        return preprocess_function(ss)
    tokenized = {}
    input_ids = []
    attention_mask = []
    
    if type(ss).__name__ == "Batch":
        ss_pd = pandas.DataFrame.from_dict(ss.data)
        ss = ss_pd.transpose()
    else:
        ss = ss.pa_table.to_pandas().transpose()
    l = len(ss.columns)
    i = 0
    input_ids_concat = []
    attention_mask_concat = []
    num_text_templates = len(text_templates)
    tt = 0
    eos = tokenizer.eos_token
    sep = eos
    while i < l:
        s = ss[i]
        i += 1
        text = eval(f'f"{text_templates[tt]}"')
        if "'conversation'" in text_templates[tt]:
            text = text.replace("\nA:", f"\n{sep}A:")
            text = text.replace("\nB:", f"\n{sep}B:")
        tt += 1
        if tt >= num_text_templates:
            tt = 0

        text = wikitext_detokenizer(text)
        text = ftfy.fix_text(text, normalization='NFKC')
        if softembeddings or PrefixTuning:
            encode_len, input_ids_sub, attention_mask_sub = tokenize_string(text)
            input_ids.append(input_ids_sub)
            attention_mask.append(attention_mask_sub)
        else:
            encode_len, input_ids_txt, attention_mask_txt = tokenize_string(text)
            input_ids_concat += input_ids_txt[:encode_len+1]
            attention_mask_concat += attention_mask_txt[:encode_len+1]
            if len(input_ids_concat) < max_input_length:
                continue
            
            while True:
                if len(input_ids_concat) > 0:
                    input_ids_part = input_ids_concat[:max_input_length]
                    attention_mask_part = attention_mask_concat[:max_input_length]
                    if len(input_ids_part) < max_input_length:
                        input_ids_part = (input_ids_part + max_input_length * [tokenizer.pad_token_id])[:max_input_length]
                        attention_mask_part = (attention_mask_part + max_input_length * [0])[:max_input_length]
                    input_ids.append(input_ids_part)
                    attention_mask.append(attention_mask_part)
                    input_ids_concat = input_ids_concat[max_input_length:]
                    attention_mask_concat = attention_mask_concat[max_input_length:]
                else:
                    break
                                
    tokenized['input_ids'] = input_ids
    tokenized['attention_mask'] = attention_mask
    if PrefixTuning:
        tokenized['labels'] = tokenized['input_ids']
    return tokenized

def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string

def preprocess_dataset(source, rate, dss, tokenize: bool = True):
    print("****************************")
    print(f"processing data: {source}")
    use_data_cache_file = True 
    if PrefixTuning:
        use_data_cache_file = False
    if rate <= 1.0:
        val_size = int(validation_data_size * rate)
    else:
        val_size = int(rate / 100)
    if val_size < 1:
        val_size = 1
    if len(dss) > 1:
        ds = dss[0]
        ds = ds.train_test_split(val_size)
        dss[0] = ds["train"]
        ds_eval = ds["test"]
        columns = ds_eval.column_names
        cache_file = f"./{cache_folder_name}/{source}_{name_to_filename(tokenizer_name)}_{training_size}_{max_input_length}_eval.cache"
        ds_eval = ds_eval.map(tokenizing_sample, batched=True, remove_columns=columns, cache_file_name=cache_file)
        if training_size > 0:
            ds_train = ds[0].select(range(training_size))
        else:
            datasets = []
            for i, ds in enumerate(dss):
                if tokenize:
                    cache_file = f"./{cache_folder_name}/{source}_{i}_{name_to_filename(tokenizer_name)}_{training_size}_{max_input_length}.cache"
                    accelerator.print("tokninzing...", cache_file)
                    ds = ds.map(tokenizing_sample, batched=True, remove_columns=columns, num_proc=5, cache_file_name=cache_file, load_from_cache_file=use_data_cache_file)
                datasets.append(ds)
            ds_train = concatenate_datasets(datasets)
    else:
        ds = dss["train"]
        ds = ds.train_test_split(val_size)
        ds_train = ds["train"]
        ds_eval = ds["test"]
        if training_size > 0:
            ds_train = ds_train.select(range(training_size))
        if tokenize:
            cache_file = f"./{cache_folder_name}/{source}_{name_to_filename(tokenizer_name)}_{training_size}_{max_input_length}.cache"
            cache_file_eval = f"./{cache_folder_name}/{source}_{name_to_filename(tokenizer_name)}_{training_size}_{max_input_length}_eval.cache"
            accelerator.print("tokninzing...", cache_file)
            columns = ds_train.column_names
            ds_eval = ds_eval.map(tokenizing_sample, batched=True, remove_columns=columns, cache_file_name=cache_file_eval)
            ds_train = ds_train.map(tokenizing_sample, batched=True, remove_columns=columns, num_proc=5, cache_file_name=cache_file, load_from_cache_file=use_data_cache_file)

    if rate < 1.0:
        ds_train = ds_train.shuffle().train_test_split(test_size=(1.0 - rate))["train"]
    elif rate > 1.0:
        if len(ds_train) > rate:
            ds_train = ds_train.shuffle().select(range(int(rate)))

    accelerator.print("**********************************************")
    accelerator.print(f'train dataset len, {source}: ', len(ds_train))
    accelerator.print(f'eval  dataset len, {source}: ', len(ds_eval))
    return ds_eval, ds_train

def get_cc100(n):
    ds = load_dataset(
        f"lcw99/cc100-ko-only-{n}-of-5", 
        split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)],
        # download_mode='force_redownload'
    )
    text_templates = ["{s['text']}"]
    source = f"cc100-{n}"
    return ds, source, text_templates

def get_dataset(tokenize):
    global text_templates, validation_data_size, train_dataset_size
    data_server = os.environ['AI_DATA_SERVER']
    # data_server = "/home/chang/hd3t/dataset/text/"
    accelerator.print("reading dataset...", dataset_source)
    dss_eval = []
    dss_train = []    
    text_templates_qna = [
        "아래 지문을 보고 질문에 답 하시오.\\n지문:{eos}{s['context']}\\n{eos}질문:{s['question']}\\n답변:{s['answer']}",
        "B: {s['context']}\\n{eos}위 글을 보고 아래 질문에 답해줘.\\n{s['question']}?\\nA: {s['answer']}",
        "B: {s['context']}\\n{eos}{s['question']}?\\nA: {s['answer']}",
    ]
    text_templates_qna2 = [
        "B: {s['title']}에 대한 질문이다. {s['question']}\\nA: {s['answer']}",
    ]
    text_templates_qna_alpaca = [
        "B: {s['instruction_kr']}\\n{s['input_kr']}\\nA: {s['output_kr']}",
    ]
    text_templates_alpaca_ko_to_en = [
        "한글원문:{eos}{s['instruction_kr']}\\n{s['input_kr']}\\n{s['output_kr']}\\n{eos}영어번역:{s['instruction']}\\n{s['input']}\\n{s['output']}\\n",
    ]
    text_templates_alpaca_en_to_ko = [
        "English:{eos}{s['instruction']}\\n{s['input']}\\n{s['output']}\\n{eos}Korean:{s['instruction_kr']}\\n{s['input_kr']}\\n{s['output_kr']}\\n",
    ]
    
    text_templates_conversation = [
        "아래 대화를 연결해 보시오.\\n{s['conversation']}",
        "아래 대화를 잘 보고 다음 대화를 연결해 나가봐.\\n{s['conversation']}",
        "아래 대화를 계속 진행 해봐..\\n{s['conversation']}",
        "아래 대화를 보고 적절한 다음 응답을 하시오.\\n{s['conversation']}",
        "대화를 계속 진행 하시오.\\n{s['conversation']}",
        "아래 대화를 계속 진행 하시오.\\n{s['conversation']}",
    ]
    text_templates_tran_ko_to_en = [
        "한글원문:{eos}{s['korean']}\\n{eos}영어번역:{s['english']}",
        "{s['korean']}\\n{eos}위글을 영어로 번역 하시오.\\n{s['english']}",
        "{s['korean']}\\n{eos}영어로 번역 하시오.\\n{s['english']}",
        "B: {s['korean']}{eos} 영어로 번역 해 줘.\\nA: {s['english']}",
        "B: {s['korean']}{eos} 영어로 번역 해 주세요.\\nA: {s['english']}",
        "B: {s['korean']}\\n{eos}A: 영어로 번역.\\nB: {s['english']}",
        "B: {s['korean']}\\n{eos}A: 영어로.\\nB: {s['english']}",
        "B: {s['korean']}\\n{eos}A: 영어로 해봐.\\nB: {s['english']}",
    ]
    text_templates_tran_en_to_ko = [
        "영어원문:{s['english']}\\n{eos}한글번역:{s['korean']}",
        "{s['english']}\\n{eos}위글을 한글로 번역 하시오.\\n{s['korean']}",
        "{s['english']}\\n{eos}한글로 번역 하시오.\\n{s['korean']}",
        "B: {s['english']}{eos} 한글로 번역 해 줘.\\nA: n{s['korean']}",
        "B: {s['english']}{eos} 한글로 번역 해 주세요.\\nA: {s['korean']}",
        "B: {s['english']}\\n{eos}A: 한글로\\nB: {s['korean']}",
        "B: {s['english']}\\n{eos}A: 한글로 번역.\\nB: {s['korean']}",
        "B: {s['english']}\\n{eos}A: 한글로 해봐.\\nB: {s['korean']}",
    ]
    text_templates_gsm8k_ko_to_en = [
        "한글원문:{eos}{s['question_kr']}\\n{s['reasoning_kr']}\\n{eos}영어번역:{s['question']}\\n{s['reasoning']}",
        "{s['question_kr']}\\n{s['reasoning_kr']}\\n{eos}위글을 영어로 번역 하시오.\\n{s['question']}\\n{s['reasoning']}",
        "{s['question_kr']}\\n{s['reasoning_kr']}\\n{eos}영어로 번역 하시오.\\n{s['question']}\\n{s['reasoning']}",
        "B: {s['question_kr']}\\n{s['reasoning_kr']}{eos} 영어로 번역 하시오.\\nA: {s['question']}\\n{s['reasoning']}",
        "B: {s['question_kr']}\\n{s['reasoning_kr']}{eos} 영어로\\nA: {s['question']}\\n{s['reasoning']}",
    ]
    text_templates_gsm8k_en_to_ko = [
        "영어원문:{s['question']}\\n{s['reasoning']}\\n{eos}한글번역:{s['question_kr']}\\n{s['reasoning_kr']}",
        "{s['question']}\\n{s['reasoning']}\\n{eos}위글을 한글로 번역 하시오.\\n{s['question_kr']}\\n{s['reasoning_kr']}",
        "{s['question']}\\n{s['reasoning']}\\n{eos}한글로 번역 하시오.\\n{s['question_kr']}\\n{s['reasoning_kr']}",
        "B: {s['question']}\\n{s['reasoning']}\\n{eos}한글로 번역 하시오.\\nA: {s['question_kr']}\\n{s['reasoning_kr']}",
        "B: {s['question']}\\n{s['reasoning']}\\n{eos}한글로\\nA: {s['question_kr']}\\n{s['reasoning_kr']}",
    ]
    text_templates_summarize = [
        "B: {s['passage']}{eos} 이걸 요약 하면?\\nA: {s['summary1']}",
        "B: {s['passage']}\\n{eos}요약 해봐.\\nA: {s['summary1']}",
        "B: 아래글을 요약 해봐.\\n{s['passage']}\\n{eos}A: {s['summary1']}",
        "A: {s['passage']}\\nB: 요약 해줘.\\nA: {s['summary1']}",
        "A: {s['passage']}\\nB: 요약 해봐.\\nA: {s['summary1']}",
        "A: {s['passage']}\\nB: 요약 하시오.\\nA: {s['summary1']}",
    ]
    text_templates_reasoning = [
        "질문에 답 하고 이유를 설명하시오.\\n{s['question_kr']}\\n{eos}정답은 {s['answer_kr']} 이고, 정답을 도출하는 과정은 다음과 같습니다.\\n{s['reasoning_kr']}",
        "질문에 답 하고 정답을 도출하는 과정을 설명하시오.\\n{s['question_kr']}\\n{eos}정답은 {s['answer_kr']} 이고, 정답을 도출하는 과정은 다음과 같습니다.\\n{s['reasoning_kr']}",
        "B: {s['question_kr']}\\n{eos}A: {s['reasoning_kr']}. 그러므로 정답은 {s['answer_kr']}",
        "B: {s['question_kr']}\\n{eos}A: 정답은 다음과 같이 도출 가능합니다.\\n{s['reasoning_kr']}\\n그러므로 정답은 {s['answer_kr']} 입니다.",
    ]
    text_templates_reasoning_softembeddings = [
        "질문에 답 하고 이유를 설명하시오.\\n{eos}{s['question_kr']}\\n{eos}정답은 {s['answer_kr']} 이고, 정답을 도출하는 과정은 다음과 같습니다.\\n{s['reasoning_kr']}",
    ]
    text_templates_reasoning_en = [
        "질문에 답 하고 이유를 설명하시오.\\n{s['question']}\\n{eos}정답은 {s['answer']} 이고, 정답을 도출하는 과정은 다음과 같습니다.\\n{s['reasoning']}",
        "질문에 답 하고 정답을 도출하는 과정을 설명하시오.\\n{s['question']}\\n{eos}정답은 {s['answer']} 이고, 정답을 도출하는 과정은 다음과 같습니다.\\n{s['reasoning']}",
        "질문에 답 하시오.\\n{s['question']}\\n{eos}정답은 {s['answer']} 이고, 정답을 도출하는 과정은 다음과 같습니다.\\n{s['reasoning']}",
        "B: {s['question']}\\n{eos}A: 정답은 {s['answer']} 이고, 정답을 도출하는 과정은 다음과 같습니다.\\n{s['reasoning']}",
        "B: {s['question']}\\n{eos}A: 정답은 다음과 같이 도출 가능합니다.\\n{s['reasoning']}\\n그러므로 정답은 {s['answer']} 입니다.",
    ]
    text_templates_news_writing = [
        "{s['title']}\\n위 문장을 주제로 신문 기사를 작성 하시오.\\n{eos}{s['text']}",
        "{s['title']}\\n위 내용을 제목으로 신문 기사를 작성 하시오.\\n{eos}{s['text']}",
        "아래 문장을 제목으로 신문 기사를 작성 하시오.\\n{s['title']}\\n{eos}{s['text']}",
        "아래 문장을 주제로 신문 기사를 작성 하시오.\\n{s['title']}\\n{eos}{s['text']}",
        "B: {s['title']}\\n위 내용을 포함하는 신문 기사를 작성 해봐.\\nA: {s['text']}",
        "B: 아래 내용을 포함하는 신문 기사를 작성 하시오.\\n{s['title']}\\nA: {s['text']}",
        "B: {s['title']} 이 내용으로 신문 기사를 써봐.\\nA: {s['text']}",
        "B: {s['title']} 이걸 제목으로 신문 기사를 써봐.\\nA: {s['text']}",
        "B: {s['title']} 이걸 제목으로 블로그를 써봐.\\nA: {s['text']}",
        "B: {s['title']} 이 내용으로 블로그를 써봐.\\nA: {s['text']}",
    ]
    
    if "sns" in dataset_source.keys():
        ds = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/한국어 SNS/korean_sns_training_gpt2_v2.json")
        text_templates = ["{s['sample']}"]
        source = "sns"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "wiki" in dataset_source.keys():
        ds = load_dataset("lcw99/wikipedia-korean-20221001")
        text_templates = ["{s['text']}"]
        source = "wiki"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "cc100-1" in dataset_source.keys():
        ds, source, text_templates = get_cc100(1)
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "cc100-2" in dataset_source.keys():
        ds, source, text_templates = get_cc100(2)
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "cc100-3" in dataset_source.keys():
        ds, source, text_templates = get_cc100(3)
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "cc100-4" in dataset_source.keys():
        ds, source, text_templates = get_cc100(4)
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "cc100-5" in dataset_source.keys():
        ds, source, text_templates = get_cc100(5)
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "oscar" in dataset_source.keys(): 
        # ds = load_dataset("oscar", language="ko", split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])
        ds = load_dataset("lcw99/oscar-ko-only", split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])
        # ds = load_dataset("oscar", language="ko")
        # ds.push_to_hub("oscar-ko-only")
        text_templates = ["{s['text']}"]
        source = "oscar"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "namu" in dataset_source.keys():
        ds = load_dataset("heegyu/namuwiki-extracted")
        text_templates = ["{s['text']}"]
        source = "namu"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "nikl_news" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}NIKL_NEWSPAPER_2021_v1.0.1.zip"})
        text_templates = ["{s['text']}"]
        source = "nikl_news"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "nikl_news_2020" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}NIKL_NEWSPAPER_2020_v1.1.1.zip"})
        text_templates = ["{s['text']}"]
        source = "nikl_news_2020"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "nikl_written" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}NIKL_WRITTEN_v1.2.zip"})
        text_templates = ["{s['text']}"]
        source = "nikl_written"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_paper_summary" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_paper_summary.zip"})
        text_templates = ["{s['entire_org']}"]
        source = "aihub_paper_summary"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_patent_summary" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_patent_summary.zip"})
        text_templates = ["{s['entire_org']}"]
        source = "aihub_patent_summary"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "tbsm" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}tbsm.json.zip"})
        text_templates = ["{s['text']}"]
        source = "tbsm"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "todays_fortune" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}todays_fortune.zip"})
        text_templates = ["{s['source']}\\n오늘의 운세:{s['target']}"]
        source = "todays_fortune"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "wikiqna" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_wiki_qna.zip"})
        text_templates = text_templates_qna
        source = "wikiqna"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "wikiqna2" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_wiki_qna.zip"})
        text_templates = text_templates_qna2
        source = "wikiqna2"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_news_qna" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_news_qna.zip"})
        text_templates = text_templates_qna
        source = "aihub_news_qna"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_book_qna" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_book_qna.zip"})
        text_templates = text_templates_qna
        source = "aihub_book_qna"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "gsm8k_train" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}gsm8k_train.zip"})
        text_templates = text_templates_reasoning
        source = "gsm8k_train"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "gsm8k_train_softembeddings" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}gsm8k_train.zip"})
        text_templates = text_templates_reasoning_softembeddings
        source = "gsm8k_train_softembeddings"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "gsm8k_train_en" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}gsm8k_train.zip"})
        text_templates = text_templates_reasoning_en
        source = "gsm8k_train_en"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_summary" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}korean_text_summary.zip"})
        text_templates = text_templates_summarize
        source = "aihub_summary"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_translation_to_english" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_translation.zip"})
        text_templates = text_templates_tran_ko_to_en
        source = "aihub_translation_to_english"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_translation_to_korean" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_translation.zip"})
        text_templates = text_templates_tran_en_to_ko
        source = "aihub_translation_to_korean"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_tech_domain_translation_to_english" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_tech_domain_translation.zip"})
        text_templates = text_templates_tran_ko_to_en
        source = "aihub_tech_domain_translation_to_english"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_tech_domain_translation_to_korean" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_tech_domain_translation.zip"})
        text_templates = text_templates_tran_en_to_ko
        source = "aihub_tech_domain_translation_to_korean"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "gsm8k_ko_to_en" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}gsm8k_train.zip"})
        text_templates = text_templates_gsm8k_ko_to_en
        source = "gsm8k_ko_to_en"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "gsm8k_en_to_ko" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}gsm8k_train.zip"})
        text_templates = text_templates_gsm8k_en_to_ko
        source = "gsm8k_en_to_ko"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_daily_conversation" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_daily_conversation.zip"})
        text_templates = text_templates_conversation
        source = "aihub_daily_conversation"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_domain_conversation" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_domain_conversation.zip"})
        text_templates = text_templates_conversation
        source = "aihub_domain_conversation"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "nikl_news_writing" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}NIKL_NEWSPAPER_2021_v1.0.1.zip"})
        text_templates = text_templates_news_writing
        source = "nikl_news_writing"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)
    if "nikl_news_2020_writing" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}NIKL_NEWSPAPER_2020_v1.1.1.zip"})
        text_templates = text_templates_news_writing
        source = "nikl_news_2020_writing"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_news_qna_writing" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_news_qna_writing.zip"})
        text_templates = text_templates_news_writing
        source = "aihub_news_qna_writing"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_administrative_documents_qna" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_administrative_documents_qna.zip"})
        text_templates = text_templates_qna
        source = "aihub_administrative_documents_qna"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_technology_science_translation_to_english" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_technology_science_translation.zip"})
        text_templates = text_templates_tran_ko_to_en
        source = "aihub_technology_science_translation_to_english"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_technology_science_translation_to_korean" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_technology_science_translation.zip"})
        text_templates = text_templates_tran_en_to_ko
        source = "aihub_technology_science_translation_to_korean"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_social_science_translation_to_english" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_social_science_translation.zip"})
        text_templates = text_templates_tran_ko_to_en
        source = "aihub_social_science_translation_to_english"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "aihub_social_science_translation_to_korean" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}aihub_social_science_translation.zip"})
        text_templates = text_templates_tran_en_to_ko
        source = "aihub_social_science_translation_to_korean"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "korquad_2.1" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}korquad_2.1.zip"})
        text_templates = text_templates_qna2
        source = "korquad_2.1"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "korquad_2.1_dev" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}korquad_2.1_dev.zip"})
        text_templates = text_templates_qna2
        source = "korquad_2.1_dev"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "alpaca" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}alpaca_data_kr_checked.zip"})
        text_templates = text_templates_qna_alpaca
        source = "alpaca"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "alpaca_ko_to_en" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}alpaca_data_kr_checked.zip"})
        text_templates = text_templates_alpaca_ko_to_en
        source = "alpaca_ko_to_en"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "alpaca_en_to_ko" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}alpaca_data_kr_checked.zip"})
        text_templates = text_templates_alpaca_en_to_ko
        source = "alpaca_en_to_ko"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "chatdoctor5k" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}chatdoctor5k_kr.zip"})
        text_templates = text_templates_qna_alpaca
        source = "chatdoctor5k"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
    if "tarot_conv" in dataset_source.keys():
        ds = load_dataset("json", data_files={'train': f"{data_server}tarot_conv_text.zip"})
        text_templates = ["{s['text']}"]
        source = "tarot_conv"
        ds_eval, ds_train = preprocess_dataset(source, dataset_source[source], ds, tokenize)
        dss_eval.append(ds_eval)
        dss_train.append(ds_train)        
                
    ds_concat_eval = concatenate_datasets(dss_eval) 
    ds_concat_train = concatenate_datasets(dss_train)
         
    accelerator.print(f'ds_concat_eval={ds_concat_eval}')
    accelerator.print(f'ds_concat_train={ds_concat_train}')
    if len(ds_concat_eval) < validation_data_size:
        validation_data_size = len(ds_concat_eval) 
    ds_eval = ds_concat_eval.shuffle().select(range(validation_data_size))
    if len(ds_concat_train) > 1024 * 10:
        train_dataset_size = int(len(ds_concat_train) / 1024) * 1024
    else:
        train_dataset_size = len(ds_concat_train)
    ds_train = ds_concat_train.shuffle().select(range(train_dataset_size))
    accelerator.print(f'combined train dataset len: ', "{:,}".format(len(ds_train)))
    return ds_eval, ds_train, text_templates
    
text_templates = None
glo_tokenize = None
# def my_collate(batch):
#     data = [item[text_templates] for item in batch]
#     if glo_tokenize:
#         data = tokenizer(data, max_length=max_input_length, truncation=True, padding=True)
#     return [data]
    
def get_dataloaders(tokenize: bool = False, loader_batch_size: int = batch_size, def_data_collator = False):
    global text_templates, glo_tokenize
    glo_tokenize = tokenize
    eval_dataset, train_dataset, text_templates = get_dataset(tokenize)
    accelerator.print(train_dataset)
    accelerator.print(eval_dataset)
    if def_data_collator:
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=loader_batch_size, collate_fn=default_data_collator)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=loader_batch_size, collate_fn=default_data_collator)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=loader_batch_size)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=loader_batch_size)
    return train_dataloader, eval_dataloader

def build_tokenizer():
    accelerator.print("\n-----------------------\ntokenizer name = ", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    accelerator.print(f"{tokenizer.pad_token=}\n{tokenizer.eos_token=}")
    accelerator.print(f"{tokenizer.pad_token_id=}\n{tokenizer.eos_token_id=}")
    tokenizer.pad_token = tokenizer.eos_token
    accelerator.print(f"pad token setted = {tokenizer.pad_token_id=}\n{tokenizer.eos_token_id=}")
    return tokenizer

def build_adam8bit_optimizer(model):
    training_args = TrainingArguments(per_device_train_batch_size=batch_size, output_dir=".")

    decay_parameters = ["norm", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    optimizer_kwargs["lr"] = 0.0006
    adam_bnb_optim = Adam8bit(
        optimizer_grouped_parameters,
        **optimizer_kwargs
    )
    return adam_bnb_optim
        
def list_model_children(model):
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if name == 'lm_head':
                    #child.out_features = 91238
                    accelerator.print("\n********************\nchild.out_features=", child.out_features, child.in_features)
                    accelerator.print(f'name = {name}, child = {child}, child.weight.shape = {child.weight.shape}')
            elif isinstance(child, nn.Embedding):
                #child.num_embeddings = 91238
                accelerator.print("\n********************\nchild.num_embeddings=", child.num_embeddings, child.embedding_dim)
                accelerator.print(f'name = {name}, child = {child}, child.weight.shape = {child.weight.shape}')
        
def unfreeze_transformer_layer(model, unfreeze_layers):
    for name, param in model.named_parameters():
        if name in unfreeze_layers:
            param.requires_grad = True      
        else:
            param.requires_grad = False      
            
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        accelerator.print(f'"{name}", = {param.requires_grad}')
    if all_param > 0:
        accelerator.print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

from megatron.model import SoftEmbedding, SoftEmbedding2                                 
def init_model():
    global start_model_path
    kwarg = {
        "torch_dtype": torch.float16,
    }
            
    if gpt_neo is not None:
        if model_file is None:
            model = f"EleutherAI/{gpt_neo}"
        else:
            accelerator.print("loading weight from file=", model_file)
            model = model_file
        if model_file != None and ".json" in model_file:
            accelerator.print("loading model-", model_file)
            gpt_config = AutoConfig.from_pretrained(model_file)
            gpt = GPTNeoForCausalLM(gpt_config)
            model = None
        else:
            accelerator.print("loading model-", model, kwarg)
            gpt = GPTNeoForCausalLM.from_pretrained(model, **kwarg)
        accelerator.print(gpt)
    else:
        accelerator.print("loading weight from file=", model_file)
        model = model_file
        # kwarg["revision"] = "float16"
        accelerator.print("loading model-", model, kwarg)
        gpt = AutoModelForCausalLM.from_pretrained(model, **kwarg)
        
        if LoRa:
            accelerator.print("LoRa enabled.......")
            target_modules = ["query_key_value", "xxx"]  # workaround to use 8bit training on this model
            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=target_modules 
            )
            # peft_config = LoraConfig(
            #     task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            # )        
            gpt = get_peft_model(gpt, peft_config)
            for name, param in gpt.named_parameters():
                if "embed_out" in name:
                    param.requires_grad = True      # just temporary patch for 'None of the inputs have requires_grad' error
            gpt.print_trainable_parameters()
        elif PrefixTuning:
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens = max_input_length)            
            gpt = get_peft_model(gpt, peft_config)
            gpt.print_trainable_parameters()
            
    start_model_path = model
    # list_model_children(gpt)
    
    if kor_voca_extention:
        tokenizer_len = len(tokenizer)
        accelerator.print("\n\n\n=====\ntokenizer_len=", tokenizer_len)
        gpt.resize_token_embeddings(tokenizer_len)
        accelerator.print("resize done....")

    if softembeddings:
        # soft_prompt = SoftEmbedding(
        #     tokenizer,
        #     gpt.gpt_neox.config.hidden_size,
        #     2048, 
        #     wte=gpt.embed_out.weight,
        #     n_tokens = 10,
        #     init_string = "",
        #     init_range = 0.5 )
        # gpt.insert_layers(
        #     layers=soft_prompt, idx=1
        # )  

        # freeze everything but the soft prompt
        s_wte = SoftEmbedding2(gpt.get_input_embeddings(), 
                      n_tokens=10, 
                      initialize_from_vocab=True)
        gpt.set_input_embeddings(s_wte)
        for name, param in gpt.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False
    elif LoRa or PrefixTuning:
        pass
    else:
        if scratch:
            if not optimizer_8bit:
                if reset_weight:
                    gpt.init_weights()  # from scarch
            # for param in gpt.base_model.parameters():
            #     if param.dtype == torch.int8:
            #         param.has_fp16_weight = True    # for training
            #         param.memory_efficient_backward = True
            #         # param.requires_grad = True      # not working now
            #     else:
            #         param.requires_grad = True    
        else:
            if len(unfreeze) > 0: 
                unfreeze_transformer_layer(gpt, unfreeze)           
    print_trainable_parameters(gpt)
    gpt.gradient_checkpointing_enable()
    
    gpt.config.__dict__["_name_or_path"] = f"{new_model_name}"
    gpt.config.__dict__["use_cache"] = False
    # gpt.save_pretrained("./StockModels/gpt-j-6B-fp16-ko-voc-saved-as-8bit")
        
    return gpt
              
def _get_lr(param_group, param_state):
    step = param_state["step"]
    eps = param_group["eps"]
    return 5e-5 - eps * step
              
def loss_function(output, input):
    # loss_cross_entropy = nn.CrossEntropyLoss()
    # loss = loss_cross_entropy(output.logits[:, :-1, :].flatten(0, -2), input['input_ids'][:, 1:].flatten()) 
    # return loss
    # else:
    # if output.loss:
    #     return output.loss
    loss = F.cross_entropy(output.logits[:, :-1, :].flatten(0, -2), input['input_ids'][:, 1:].flatten(), reduction='mean')
    return loss

def petf_trainer():
    global batch_size, train_dataloader, eval_dataloader

    model = init_model()
    # model = accelerator.prepare(
    #     model
    # )

    train_dataloader, eval_dataloader = get_dataloaders(tokenize=True, loader_batch_size=batch_size)
    if data_build_only:
        return
    
    num_training_steps = len(train_dataloader.dataset)
    max_steps = -1


    if optimizer_8bit:
        optimizer = Adam8bit(model.parameters(), lr=0.0006, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 100, num_training_steps
        )
    else:
        optimizer_cls = (
            torch.optim.AdamW
            if accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else accelerate.utils.DummyOptim
        )
        
        optimizer = optimizer_cls(model.parameters(), lr=0.0006)
        lr_scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=num_training_steps, warmup_num_steps=300)

    accelerator.register_for_checkpointing(lr_scheduler)
 
    # lr_scheduler = AdafactorSchedule(optimizer)    
    # optimizer._get_lr = _get_lr
    
    # model = model.to(device)
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
    )
    
    for epoch in range(1):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch1 = {}
            for k, vv in batch.items():
                l = []
                for v in vv:
                    # v = v.to(device)
                    l.append(v)
                batch1[k] = torch.stack(l)
            outputs = model(**batch1)
            loss = outputs.loss
            total_loss += loss.detach().float()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

        eval_epoch_loss = eval_loss/len(train_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss/len(eval_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")    
    
def trainer():
    global batch_size
    if batch_size == 0:
        batch_size = 4

    train_dataloader, eval_dataloader = get_dataloaders(tokenize=False, loader_batch_size=batch_size)
    if data_build_only:
        return

    model, optimizer = init_model()
 
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(num_training_steps*0.1), num_training_steps
    )

    # if not optimizer_8bit:
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #         model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    #     )
    
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        batch_token = tokenizer(batch, truncation=True, padding=True, max_length=max_input_length, return_tensors='pt')
        with torch.cuda.amp.autocast():
            outputs = model(**batch_token.to(device))
            loss = loss_function(outputs, batch_token)
            # loss = loss.mean()  # mean() to average on multi-gpu parallel training
        accelerator.backward(loss)
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # scaler.step(optimizer)
        # scaler.update()
                
        lr_scheduler.step()
        if i % 10 == 0 and i != 0:
            state = {
                'k' : i, 'epoch': num_epochs, 
                'lr_scheduler': lr_scheduler.state_dict(), 
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }   
            accelerator.print(loss)
            # torch.save(state, "./TestModel/model.pt")

class MyTrainer(Trainer):    
    # def create_optimizer_and_scheduler(self, num_training_steps):
    #     self.optimizer = Adam8bit(self.model.parameters(), lr=1e-5)
    #     self.lr_scheduler = AdafactorSchedule(self.optimizer)    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            loss = self.scaler.scale(loss)
            accelerator.backward(loss)
            # self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            #loss.backward()
            accelerator.backward(loss)
        # loss.backward(retain_graph=True)

        return torch.squeeze(loss.detach())

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.do_predict:
            return (None, None, None)
        
        global last_eval_model
        
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = None

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
            
        last_eval_model = model
        return (loss, logits, labels)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model
    

    loss_cross_entropy = nn.CrossEntropyLoss()
    def compute_loss(self, model, inputs, return_outputs=False):
        # return super(MyTrainer, self).compute_loss(outputs, inputs, return_outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        outputs = model(**inputs)
        # if "loss" in outputs:
        #     loss = outputs["loss"]
        # else:
        loss = self.loss_cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), inputs['input_ids'][:, 1:].flatten()) 
            # loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), inputs['input_ids'][:, 1:].flatten(),
            #                    reduction='mean')
        #print("loss=", loss)
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            dl = DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            return accelerator.prepare(dl)
        
        train_sampler = self._get_train_sampler()

        dl = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return accelerator.prepare(dl)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            dl = DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            return accelerator.prepare(dl)

        eval_sampler = self._get_eval_sampler(eval_dataset)

        dl = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return accelerator.prepare(dl)
    
metric_accuracy = evaluate.load("accuracy")
perplexity = evaluate.load("perplexity", module_type="metric")

def get_perplexity():
    return 1000.0
    try:
        data = load_from_disk("./test_data")['text']
        input_texts = [s[:1024] for s in data if s!='']
        latest_model_dir = max(glob.glob(os.path.join(model_save_dir, 'checkpoint-*/')), key=os.path.getmtime)
        accelerator.print(f"latest_model_dir for perplexity={latest_model_dir}")
        result = perplexity.compute(model_id=latest_model_dir, predictions=input_texts)
        return result['mean_perplexity']
    except Exception as e:
        accelerator.print("\n!! get_perplexity error1= ", e)
        if start_model_path is not None:
            try:
                result = perplexity.compute(model_id=start_model_path, predictions=input_texts)
                return result['mean_perplexity']
            except Exception as e:
                accelerator.print("\n!! get_perplexity error2= ", e)
                return 1000.0            
        accelerator.print("\n!! get_perplexity error= ", e)
        return 0.0

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
    
    # pred_ids = numpy.array(pred_ids)
    # labels = labels_ids.reshape(-1)
    # preds = pred_ids.reshape(-1)
    # mask = labels != -100
    # labels = labels[mask]
    # #preds = preds[mask]
    # acc = metric_accuracy.compute(predictions=preds, references=labels)
            
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)

    ppl = {}
    ppl["mean_perplexity"] = get_perplexity()

    accelerator.print("\n\n===========predictions first token\n", pred_str[0].replace('\n', '/'))

    if eval_sample:
        tt = tokenizer("It's cold now, but", max_length=max_input_length, truncation=True, return_tensors='pt').to(device)
        output_sequences = last_eval_model(tt["input_ids"])
        pred_ids = torch.argmax(output_sequences["logits"][0], dim=-1)
        generated = tokenizer.decode(pred_ids, skip_special_tokens=True)     
        generated = generated.replace('\n', '/')   
        accelerator.print(f"\n{generated}\n")
        
        tt = tokenizer("봄이 왔어요. 이제 곧", max_length=max_input_length, truncation=True, return_tensors='pt').to(device)
        output_sequences = last_eval_model(tt["input_ids"])
        pred_ids = torch.argmax(output_sequences["logits"][0], dim=-1)
        generated = tokenizer.decode(pred_ids, skip_special_tokens=True)     
        generated = generated.replace('\n', '/')   
        accelerator.print(f"\n{generated}\n")
        
    return {
        "mean_perplexity": round(ppl["mean_perplexity"], 4)
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_list = []
    for logit in logits:
        pred = torch.argmax(logit, dim=-1)
        pred_list.append(pred)
        
    try:
        batch = len(logits)
        ii = random.randint(0, batch-1)
        pred_str = tokenizer.batch_decode(pred_list[ii], skip_special_tokens=False)
        pred_str = " ".join([str(i) for i in pred_str])
        pred_str = pred_str.replace("\n", "/")
        accelerator.print(f"\n**{ii} ", pred_str)
        if len(labels) > ii:
            labels_ids = labels[ii]
            labels_ids[labels_ids == -100] = tokenizer.pad_token_id
            decoded_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
            label_str = "_".join([str(i) for i in decoded_str])
            label_str = label_str.replace("\n", "/")
            accelerator.print(f"\n=={ii} ", label_str)
    except Exception as e:
        accelerator.print("\n!! ", e, ii, len(labels))
    return pred_list, labels

def huggingface_trainer():
    global batch_size, train_dataloader, eval_dataloader

    train_dataloader, eval_dataloader = get_dataloaders(tokenize=True, loader_batch_size=batch_size)
    if data_build_only:
        return
    
    model = init_model()
    
    num_training_steps = len(train_dataloader.dataset)
    max_steps = -1

    is_ds = deepspeed_config_json is not None
    warmup_steps = 0
    if is_ds:
        warmup_steps = 200
    if train_dataset_size < (batch_size * gradient_acc) * 500:
        warmup_steps = 30

    if optimizer_8bit:
        optimizer = Adam8bit(model.parameters(), lr=0.0006, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 100, num_training_steps
        )
    else:
        optimizer_cls = (
            torch.optim.AdamW
            if accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else accelerate.utils.DummyOptim
        )
        
        optimizer = optimizer_cls(model.parameters(), lr=0.0006)
        lr_scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=num_training_steps, warmup_num_steps=warmup_steps)

    # lr_scheduler = AdafactorSchedule(optimizer)    
    # optimizer._get_lr = _get_lr

    
    if batch_size == 0:
        auto_find_batch_size = True
        batch_size = 2
    else:
        auto_find_batch_size = False
    
    args = TrainingArguments(
        model_save_dir,
        #max_steps=max_steps,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_step,
        warmup_steps=warmup_steps,
        # learning_rate=5e-5,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        auto_find_batch_size=auto_find_batch_size,
        gradient_accumulation_steps=gradient_acc,
        gradient_checkpointing=True,
        eval_accumulation_steps=gradient_acc,
        weight_decay=0.0,
        save_total_limit=5,
        num_train_epochs=num_train_epochs,
        # predict_with_generate=True,
        fp16=True if is_ds else False,
        #bf16=True,
        # fp16_backend="amp",
        fp16_full_eval=True if is_ds else False,
        load_best_model_at_end=True,
        report_to="tensorboard",
        ignore_data_skip=ignore_data_skip,     # set true for ignore batch skip, fast
        remove_unused_columns=False,
        do_predict=not skip_eval,
        do_train=True,
        deepspeed=deepspeed_config_json if is_ds else None,
        metric_for_best_model = None if skip_eval else "eval_loss"
    )
    
    print(args)
    
    hf_trainer = Trainer
    # if not optimizer_8bit:
    #     hf_trainer = MyTrainer
        
    data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="pt", mlm=False)
    trainer = hf_trainer(
        model=model,
        args=args,
        train_dataset = train_dataloader.dataset,
        eval_dataset = eval_dataloader.dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if not skip_eval else None,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )

    trainer.optimizers=(optimizer, lr_scheduler)

    # model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
    #     model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
    # )

    accelerator.print("start trainning -----------------------------")
    if continue_train:
        trainer.train(True)
    else:
        trainer.train()
    trainer.save_model()
    if LoRa:
        model.save_pretrained(f"{save_path}/final")
                                    
def main():
    global start_model_path, model_save_dir, dataset_source, tokenizer_name, max_input_length, continue_train, \
            training_size, batch_size, tokenizer, eval_sample, scratch, kor_voca_extention, optimizer_8bit, \
            unfreeze, gpt_neo, model_file, save_path, num_train_epochs, gradient_acc, \
            save_step, eval_step, validation_data_size, train_dataset_size, ignore_data_skip, reset_weight, skip_eval, \
            deepspeed_config_json, new_model_name, cache_folder_name, data_build_only, LoRa, PrefixTuning, softembeddings
    
    parser_config = argparse.ArgumentParser()
    parser_config.add_argument("--config_file", help = "loading config json file")
    
    parser = argparse.ArgumentParser(parents=[parser_config], add_help=False)
    parser.add_argument("-c", "--continue_train", action='store_true', help = "continue trainning")
    parser.add_argument("-d", "--dataset", help = "dataset source = [sns, wiki, cc100, namu]")
    parser.add_argument("-t", "--tokenizer", help = "tokenizer name")
    parser.add_argument("-i", "--max_input_length", help = "max input length")
    parser.add_argument("-s", "--training_size", help = "training size, 0 for all")
    parser.add_argument("-b", "--batch_size", help = "batch size, 0 for auto")
    parser.add_argument("--eval_sample", action='store_true', help = "eval sample")
    parser.add_argument("--scratch", action='store_true', help = "training from scratch")
    parser.add_argument("--optimizer_8bit", action='store_true', help = "load in 8bit")
    parser.add_argument("--kor_voca", action='store_true', help = "use extended kor tokenizer")
    parser.add_argument("--unfreeze", help = "set layer names to unfreeze")
    parser.add_argument("--gpt_neo", help = "gpt-neo model")
    parser.add_argument("--model_file", help = "local model file path")
    parser.add_argument("--save_path", help = "model save path")
    parser.add_argument("--num_epochs", help = "set num of epochs")
    parser.add_argument("--gradient_acc", help = "gradient accumulation")
    parser.add_argument("--save_step", help = "step for checkpoint saving")
    parser.add_argument("--eval_step", help = "step for evaluation")
    parser.add_argument("--validation_data_size", help = "validation_data_size")
    parser.add_argument("--ignore_data_skip", action='store_true', help = "ignore data skip when continue training")
    parser.add_argument("--reset_weight", action='store_true', help = "rest all weight in model")
    parser.add_argument("--skip_eval", action='store_true', help = "skip eval step")
    parser.add_argument("--deepspeed_config_json", help = "deepspeed_config_json file")
    parser.add_argument("--data_build_only", action='store_true', help = "build dataset, no training")
    parser.add_argument("--lora", action='store_true', help = "using LoRa in training")
    parser.add_argument("--prefixtuning", action='store_true', help = "using PrefixTuning in training")
    parser.add_argument("--softembeddings", action='store_true', help = "using softembeddings")

    args_config, unknown = parser_config.parse_known_args()

    if args_config.config_file:
        config = json.load(open(args_config.config_file))
        parser.set_defaults(**config)
    
    args = parser.parse_args()
    
    if args.dataset:
        dataset_source = args.dataset
    else:
        parser.print_help(sys.stderr)
        return

    if args.continue_train:
        accelerator.print("=== param continue trainning")
        continue_train = True

    if args.max_input_length:
        max_input_length = int(args.max_input_length)
    if args.training_size:
        training_size = int(args.training_size)
    if args.batch_size:
        batch_size = int(args.batch_size)
    if args.eval_sample:
        eval_sample = True
    if args.scratch:
        scratch = True
    if args.optimizer_8bit:
        optimizer_8bit = True
    if args.kor_voca:
        kor_voca_extention = True
    if args.unfreeze:
        unfreeze = args.unfreeze
    if args.gpt_neo:
        gpt_neo = args.gpt_neo
    if args.model_file:
        model_file = args.model_file
    if args.save_path:
        save_path = args.save_path
    if args.num_epochs:
        num_train_epochs = int(args.num_epochs)
    if args.gradient_acc:
        gradient_acc = int(args.gradient_acc)
    if args.save_step:
        save_step = int(args.save_step)
    if args.eval_step:
        eval_step = int(args.eval_step)
    if args.validation_data_size:
        validation_data_size = int(args.validation_data_size)
    if args.ignore_data_skip:
        ignore_data_skip = True
    if args.reset_weight:
        reset_weight = True
    if args.skip_eval:
        skip_eval = True
    if args.deepspeed_config_json:
        deepspeed_config_json = args.deepspeed_config_json
    if args.new_model_name:
        new_model_name = args.new_model_name
    if args.cache_folder_name:
        cache_folder_name = args.cache_folder_name
    if args.data_build_only:
        data_build_only = True
    if args.lora:
        LoRa = True
    if args.prefixtuning:
        PrefixTuning = True
    if args.softembeddings:
        softembeddings = True
                
    if not os.path.exists(f"./{cache_folder_name}"):
        os.makedirs(f"./{cache_folder_name}")

    if scratch:
        kor_voca_extention = False
        
    base_model_name = new_model_name
        
    # if tokenizer name provided, override previous settings.
    if args.tokenizer:
        tokenizer_name = args.tokenizer

    base_model_name += name_to_filename(tokenizer_name)
    if scratch:
        base_model_name += "_rebuild"
    else:
        base_model_name += "_fine-tune"

    accelerator.print(f"\n---------\nmodel name: {base_model_name}")
        
    model_name = f'{base_model_name}'
    
    if save_path is None:
        model_save_dir = f"./Models/{model_name}"
    else:
        model_save_dir = save_path
        
    tokenizer = build_tokenizer()
    
    if PrefixTuning:
        petf_trainer()
        # huggingface_trainer()
    else:
        huggingface_trainer()
    
if __name__ == '__main__':
    sys.exit(main()) 