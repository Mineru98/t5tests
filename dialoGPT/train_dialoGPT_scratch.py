
"""## Load the dataset"""

import transformers
from datasets import load_dataset, load_metric
import evaluate
import random, pickle, os

step_factor = 10
#medium_datasets = load_dataset("json", data_files="test_data.json")
medium_datasets = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/한국어 SNS/korean_sns_training_gpt2_v2.json")

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
    medium_datasets["train"] = medium_datasets["train"].shuffle().select(range(11000))
else:
    medium_datasets["train"] = medium_datasets["train"].shuffle()
medium_datasets["validation"] = medium_datasets["validation"].shuffle().select(range(200 * step_factor))
medium_datasets["test"] = medium_datasets["test"].shuffle().select(range(200 * step_factor))

print(medium_datasets)

"""## Data preprocessing"""

import nltk
nltk.download('punkt')
import string
from transformers import  PreTrainedTokenizerFast, AutoConfig, PretrainedConfig

model_name = "dialoGPT-medium-korean-chit-chat-scratch"

model_checkpoint = "byeongal/Ko-DialoGPT"
#model_checkpoint = "microsoft/DialoGPT-medium"
#model_checkpoint = f"./Models/{model_name}/checkpoint-158000"   # restore and continue
resume_checkpoint = f"./Models/{model_name}/checkpoint-148000"   # restore and continue

model_dir = f"./Models/{model_name}"

max_input_length = 1000
#max_target_length = 128

#tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_checkpoint)

#tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
#tokenizer = T5TokenizerFast.from_pretrained(model_dir, local_files_only=True)
#tokenizer.model_max_length = max_target_length

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
  samples = []
  for i in range(len(examples["sample"])):
    #print(i, examples["source"][i])
    #print(i, examples["target"][i])
    sample = examples["sample"][i]
    sample = sample.replace("\n", tokenizer.eos_token)
    samples.append(sample + tokenizer.eos_token)
  #texts_cleaned = [clean_text(text) for text in samples]
  #print("---->", texts_cleaned)
  #inputs = [prefix + text for text in texts_cleaned]
  #print("---->", inputs)
  inputs = samples
  tokenizer.pad_token = tokenizer.eos_token
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)
  #print("input)_ids len = ", len(model_inputs["input_ids"]))

  # Setup the tokenizer for targets
  # print("TT---->", examples["target"])
  # with tokenizer.as_target_tokenizer():
  #   labels = tokenizer(examples["target"], max_length=max_target_length, 
  #                      truncation=True)

  # model_inputs["labels"] = model_inputs["input_ids"]
  # print("model_inputs", model_inputs)
  return model_inputs

#medium_datasets_cleaned = medium_datasets.filter(lambda example: (len(example['sample']) >= 100))
medium_datasets_cleaned = medium_datasets
train_data_len = len(medium_datasets_cleaned["train"])
print("no_train_data(filterd)=", train_data_len)

tokenized_datasets = medium_datasets_cleaned.map(preprocess_data, batched=True, 
    cache_file_names=["train", "val", "test"], load_from_cache_file=True)

print(tokenized_datasets)

"""## Fine-tune T5"""

from transformers import AutoModelWithLMHead, TrainingArguments, \
                          DataCollatorForLanguageModeling, Trainer, GPT2LMHeadModel

#!rm -r {model_dir}

batch_size = 8
args = TrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=50 * step_factor,
    logging_strategy="steps",
    logging_steps=50 * step_factor,
    save_strategy="steps",
    save_steps=100 * step_factor,
    learning_rate=5e-5,
    #per_device_train_batch_size=batch_size,
    #per_device_eval_batch_size=batch_size,
    auto_find_batch_size=True,
    weight_decay=0.000001,
    save_total_limit=20,
    num_train_epochs=1,
    #predict_with_generate=True,
    fp16=True,
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

def compute_metrics_rouge(pred):
    # print("pred=", np.array(pred.predictions, dtype=object).shape)
    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    # predictions = np.argmax(pred.predictions[0], axis=-1)
    # output = metric_accuracy.compute(predictions=predictions, references=labels_ids)
    # print("output=", output)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    try:
        pred_str_filterd = [s for s in pred_str if len(s) > 0]
        print("----------pridiction=", pred_str_filterd[:100])
        ppl = perplexity.compute(predictions=pred_str_filterd, model_id='gpt2')
        #print("******************ppl=", ppl)
        random.shuffle(pred_str_filterd)
        print("===========pridiction=", pred_str_filterd[:100])
    except Exception as e:
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$ ppl error=", e)
      ppl["mean_perplexity"] = 0.0

    #print("pred_str=", pred_str)
    #print("label_str=", label_str)
    # rouge_output = rouge.compute(
    #     predictions=pred_str,
    #     references=label_str,
    #     rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    # )
    #print("rouge_output=", rouge_output)
    #return rouge_output
    #return {k: v for k, v in rouge_output.items()}

    #loss = model(input_ids=pred_ids, labels=labels_ids).loss
    #print("loss=", loss)
    # return {
    #     "R1_recall": round(rouge_output["rouge1"].mid.recall, 4),
    #     "R1_precision": round(rouge_output["rouge1"].mid.precision, 4),
    #     "R1_fmeasure": round(rouge_output["rouge1"].mid.fmeasure, 4),
    #     "R2_recall": round(rouge_output["rouge2"].mid.recall, 4),
    #     "R2_precision": round(rouge_output["rouge2"].mid.precision, 4),
    #     "R2_fmeasure": round(rouge_output["rouge2"].mid.fmeasure, 4),
    #     "RL_recall": round(rouge_output["rougeL"].mid.recall, 4),
    #     "RL_precision": round(rouge_output["rougeL"].mid.precision, 4),
    #     "RL_fmeasure": round(rouge_output["rougeL"].mid.fmeasure, 4),
    #     "RLS_recall": round(rouge_output["rougeLsum"].mid.recall, 4),
    #     "RLS_precision": round(rouge_output["rougeLsum"].mid.precision, 4),
    #     "RLS_fmeasure": round(rouge_output["rougeLsum"].mid.fmeasure, 4),
    # }
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

def compute_metrics_old(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# Function that returns an untrained model to be trained
def model_init():
    #model =  AutoModelWithLMHead.from_pretrained(model_checkpoint)
    #config = AutoConfig.from_pretrained("microsoft/DialoGPT-medium")
    config = AutoConfig.from_pretrained("./configs/config_dialogGPT_ko_medium")
    print("****config=", config)
    model = AutoModelWithLMHead.from_config(config)    
    return model
     
class MyTrainer(Trainer):     
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
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        #print("loss=", loss)
        return (loss, outputs) if return_outputs else loss

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
#trainer.train()
trainer.train(True)
#trainer.train(resume_checkpoint)

trainer.save_model()

