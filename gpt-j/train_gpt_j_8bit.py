
"""## Load the dataset"""

import transformers
from datasets import load_dataset, load_metric, load_from_disk
import evaluate
import random, pickle, os, glob

import nltk
nltk.download('punkt')
import string
from transformers import  PreTrainedTokenizerFast, AutoConfig, PretrainedConfig, AutoTokenizer

import argparse

use_weight = False
continue_train = False
fine_tune = False
model_name = "GPT-j-6B-8bit-wikipedia-finetune"

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--use_weight", help = "using weight")
parser.add_argument("-c", "--continue_train", help = "continue trainning")
parser.add_argument("-f", "--fine_tune", help = "fine tune original model")
args = parser.parse_args()
if args.use_weight:
    print("=== param using weight of model")
    use_weight = True
    model_name = model_name + "-use-weight"
if args.continue_train:
    print("=== param continue trainning")
    continue_train = True
if args.fine_tune:
    print("=== param fine tune original model")
    fine_tune = True
    model_name = "GPT-j-6B-8bit-wikipedia-finetune-org-model"
    dataset_cache_path = "./wikipedia-tokenized-org-tokenizer"
else:
    dataset_cache_path = "./wikipedia-tokenized"

print("--------------------")
print("trainning:", model_name)
print("--------------------")

if fine_tune:
    tokenizer_path = "EleutherAI/gpt-j-6B"
else:
    tokenizer_path = "../train_tokenizer/tokenizer_wikipedia_gpt_j"

step_factor = 10
model_size = "medium" # small, medium
dataset_source = "wiki" # sns, wiki
feature_name = "text" # sample, text

num_train_epochs = 2
huggingface_trainner = True


model_checkpoint = "hivemind/gpt-j-6B-8bit"

model_dir = f"./Models/{model_name}"


if dataset_source == "sns":
    medium_datasets = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/한국어 SNS/korean_sns_training_gpt2_v2.json")
else:
    medium_datasets = load_from_disk("/home/chang/nas1/linux/dataset/text/wikipedia/20221001.kr")

print(medium_datasets)

"""## Dataset train/validation/test split"""

datasets_train_test = medium_datasets["train"].train_test_split(test_size=5000)
datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=5000)

medium_datasets["train"] = datasets_train_validation["train"]
medium_datasets["validation"] = datasets_train_validation["test"]
medium_datasets["test"] = datasets_train_test["test"]

print(medium_datasets)

n_samples_train = len(medium_datasets["train"])
n_samples_validation = len(medium_datasets["validation"])
n_samples_test = len(medium_datasets["test"])
n_samples_total = n_samples_train + n_samples_validation + n_samples_test

print(f"- Training set: {n_samples_train*100/n_samples_total:.2f}%")
print(f"- Validation set: {n_samples_validation*100/n_samples_total:.2f}%")
print(f"- Test set: {n_samples_test*100/n_samples_total:.2f}%")

# keep only a subsample of the datasets
if step_factor == 1:
    medium_datasets["train"] = medium_datasets["train"].select(range(11000))
else:
    if dataset_source == "sns":
        medium_datasets["train"] = medium_datasets["train"].select(range(1000000))
    else:
        medium_datasets["train"] = medium_datasets["train"]
medium_datasets["validation"] = medium_datasets["validation"].select(range(5 * step_factor))
medium_datasets["test"] = medium_datasets["test"].select(range(5 * step_factor))

print(medium_datasets)

"""## Data preprocessing"""


max_input_length = 256
#max_target_length = 128

#tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
#tokenizer = PreTrainedTokenizerFast.from_pretrained(model_checkpoint)
#tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(f"{tokenizer_path}")

#tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
#tokenizer = T5TokenizerFast.from_pretrained(model_dir, local_files_only=True)
#tokenizer.model_max_length = max_target_length

tokenizer.pad_token = tokenizer.eos_token

prefix = ""

# def clean_text(text):
#   sentences = nltk.sent_tokenize(text.strip())
#   sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
#   sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
#                                  if len(sent) > 0 and
#                                  sent[-1] in string.punctuation]
#   text_cleaned = "\n".join(sentences_cleaned_no_titles)
#   return text_cleaned

# def preprocess_data2(examples):
#   samples = []
#   for i in range(len(examples["source"])):
#     #print(i, examples["source"][i])
#     #print(i, examples["target"][i])
#     sample = examples["source"][i] + examples["target"][i]
#     sample = sample.replace("\n", tokenizer.eos_token)
#     input_id = tokenizer.encode(sample)
#     #print(input_id)
#     samples.append(input_id[:max_input_length])
#   return samples

def preprocess_data(examples):
    print(".", end="")        
    samples = []
    for i in range(len(examples[f"{feature_name}"])):
        sample = examples[f"{feature_name}"][i]
        sample = sample.replace("\n", tokenizer.eos_token)
        samples.append(sample + tokenizer.eos_token)
    inputs = samples
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)
    return model_inputs

#medium_datasets_cleaned = medium_datasets.filter(lambda example: (len(example['sample']) >= 100))
medium_datasets_cleaned = medium_datasets
train_data_len = len(medium_datasets_cleaned["train"])
print("no_train_data(filterd)=", train_data_len)

if huggingface_trainner:
    if os.path.exists(dataset_cache_path):
        tokenized_datasets = load_from_disk(dataset_cache_path)    
    else:
        tokenized_datasets = medium_datasets_cleaned.map(preprocess_data, batched=True, load_from_cache_file=True)
        print("done loading:", tokenized_datasets)
        tokenized_datasets.save_to_disk(dataset_cache_path)

from transformers import AutoModelWithLMHead, TrainingArguments, \
                          DataCollatorForLanguageModeling, Trainer, GPTJConfig

#!rm -r {model_dir}

batch_size = 4
args = TrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    learning_rate=5e-5,
    #per_device_train_batch_size=batch_size,
    #per_device_eval_batch_size=batch_size,
    auto_find_batch_size=True,
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

data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="pt", mlm=False)

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#rouge = load_metric("rouge")
metric_accuracy = evaluate.load("accuracy")
perplexity = evaluate.load("perplexity", module_type="metric")

def compute_metrics_accuracy2(p):
    pred, labels = p
    print("pred=", pred)
    pred = np.argmax(pred)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

prev_model_dir = ""
prev_ppl = {}
def compute_metrics_rouge(eval_pred):
    global prev_model_dir, prev_ppl, trainer
    # print("pred=", np.array(pred.predictions, dtype=object).shape)
    labels_ids = eval_pred.label_ids
    pred_ids = eval_pred.predictions[0]

    # predictions = np.argmax(pred.predictions[0], axis=-1)
    # output = metric_accuracy.compute(predictions=predictions, references=labels_ids)
    # print("output=", output)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    ppl = {}
    ppl["mean_perplexity"] = 0.0

    for i in range(5):
        print("\n===========pridiction=", pred_str[i])
        print("===========referrence=", label_str[i])

    # if prev_model_dir != latest_model_dir:
    #     prev_model_dir = latest_model_dir
    #     ppl = perplexity.compute(predictions=pred_str_filterd, model_id=latest_model_dir)
    #     prev_ppl = ppl
    # else:
    #     ppl = prev_ppl 

    return {
        "mean_perplexity": round(ppl["mean_perplexity"], 4)
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

def compute_metrics_glue(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    output = metric.compute(predictions=predictions, references=labels)
    print(output)
    return output

def compute_metrics_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    output = metric_accuracy.compute(predictions=predictions, references=labels)
    print("metrics=", output)
    return output

from gpt_j_8bit import GPTJBlock, GPTJForCausalLM, GPTJModel, add_adapters, Adam8bit
from loguru import logger
import time, os
import torch.nn.functional as F

transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J


# Function that returns an untrained model to be trained
def model_init():
    if fine_tune:
        gpt =  GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
    else:
        # Initializing a GPT-J 6B configuration
        if continue_train:
            latest_model_dir = max(glob.glob(os.path.join(model_dir, 'checkpoint-*/')), key=os.path.getmtime)
            model =  GPTJForCausalLM.from_pretrained(latest_model_dir, low_cpu_mem_usage=True)
            #configuration = GPTJConfig()
            #model = GPTJForCausalLM(configuration)
        else:
            model =  GPTJForCausalLM.from_pretrained(model_checkpoint, low_cpu_mem_usage=True)
        if use_weight:
            gpt = model
        else:
            #configuration = GPTJConfig()
            configuration = model.config
            gpt = GPTJForCausalLM(configuration)
    gpt.config.__dict__["_name_or_path"] = "lcw99/gpt-j-6B-8bit"
    gpt.config.__dict__["use_cache"] = False
    add_adapters(gpt)
    gpt.to('cuda')
    gpt.gradient_checkpointing_enable()
    return gpt
     
class MyTrainer(Trainer):    
    # def create_optimizer_and_scheduler(self, num_training_steps):
    #     self.optimizer = Adam8bit(self.model.parameters(), lr=1e-5)

    #     self.scheduler=transformers.get_cosine_schedule_with_warmup(optimizer=self.optimizer,
    #             num_warmup_steps = 200,
    #             num_training_steps = num_training_steps)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Recursively unwraps a model from potential containers (as used in distributed training).
        Args:
            model (`torch.nn.Module`): The model to unwrap.
        """
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), inputs['input_ids'][:, 1:].flatten(),
                               reduction='mean')

        #print("loss=", loss)
        return (loss, outputs) if return_outputs else loss


if huggingface_trainner:
    trainer = MyTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_rouge,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Commented out IPython magic to ensure Python compatibility.
    # Start TensorBoard before training to monitor it in progress
    # %load_ext tensorboard
    # %tensorboard --logdir '{model_dir}'/runs

    print("start trainning -----------------------------")
    if continue_train:
        trainer.train(True)
    else:
        trainer.train()
    #trainer.train(resume_checkpoint)
    trainer.save_model()

else:

    model =  GPTJForCausalLM.from_pretrained(model_checkpoint, low_cpu_mem_usage=True)
    #configuration = GPTJConfig()

    #gpt = GPTJModel(model.config)
    gpt = model
    add_adapters(gpt)
    gpt.to('cuda')
    gpt.gradient_checkpointing_enable()

    # example dataset
    #dataset = load_dataset("transformersbook/codeparrot-train", streaming=True)

    # custom dataset
    dataset = medium_datasets_cleaned

    optimizer = Adam8bit(gpt.parameters(), lr=1e-5)

    # Set the model to training mode
    start = time.time()

    # Training loop
    with torch.cuda.amp.autocast():
        for row in tqdm(dataset["train"]):
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
            optimizer.zero_grad()


    logger.info("Finished fine-tuning in {}".format(time.time() - start))

    # --------------> Saving fine-tuned model <-----------------#
    try:
        save_dir = "./finetuned_gpt-j-8_bit"
        os.makedirs(save_dir)
        gpt.save_pretrained(save_dir)
    except Exception as e:
        #print("Error saving model: ", e)
        logger.info("Error saving model: {}".format(e))