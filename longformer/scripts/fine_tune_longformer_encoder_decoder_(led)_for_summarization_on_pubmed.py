import os

model_path = "./led_model_ko_summ"
output_dir = "./led_tune"

from datasets import load_dataset, load_metric


medium_datasets = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/022.요약문 및 레포트 생성 데이터/korean_text_summary.zip")

#train_dataset = load_dataset("scientific_papers", "pubmed", split="train")
#val_dataset = load_dataset("scientific_papers", "pubmed", split="validation")

datasets_train_test = medium_datasets["train"].train_test_split(test_size=300)

train_dataset = datasets_train_test["train"]
val_dataset = datasets_train_test["test"]

"""It's always a good idea to take a look at some data samples. Let's do that here."""

import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=4):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

"""We can see that the input data is the `article` - a scientific report and the target data is the `abstract` - a concise summary of the report.

Cool! Having downloaded the dataset, let's tokenize it.
We'll import the convenient `AutoTokenizer` class.
"""

from transformers import AutoTokenizer

""" and load the tokenizer"""

tokenizer = AutoTokenizer.from_pretrained(model_path)

"""Note that for the sake of this notebook, we finetune the "smaller" LED checkpoint ["allenai/led-base-16384"](https://huggingface.co/allenai/led-base-16384). Better performance can however be attained by finetuning ["allenai/led-large-16384"](https://huggingface.co/allenai/led-large-16384) at the cost of a higher required GPU RAM.

Pubmed's input data has a median token length of 2715 with the 90%-ile token length being 6101. The output data has a media token length of 171 with the 90%-ile token length being 352.${}^1$. 

Thus, we set the maximum input length to 8192 and the maximum output length to 512 to ensure that the model can attend to almost all input tokens is able to generate up to a large enough number of output tokens.

In this notebook, we are only able to train on `batch_size=2` to prevent out-of-memory errors.

---
${}^1$ The data is taken from page 11 of [Big Bird: Transformers for Longer Sequences](https://arxiv.org/pdf/2007.14062.pdf).
"""

max_input_length = 8192
max_output_length = 1024
batch_size = 4

"""Now, let's write down the input data processing function that will be used to map each data sample to the correct model format.
As explained earlier `article` represents here our input data and `abstract` is the target data. The datasamples are thus tokenized up to the respective maximum lengths of 8192 and 512.

In addition to the usual `attention_mask`, LED can make use of an additional `global_attention_mask` defining which input tokens are attended globally and which are attended only locally, just as it's the case of [Longformer](https://huggingface.co/transformers/model_doc/longformer.html). For more information on Longformer's self-attention, please take a look at the corresponding [docs](https://huggingface.co/transformers/model_doc/longformer.html#longformer-self-attention). For summarization, we follow recommendations of the [paper](https://arxiv.org/abs/2004.05150) and use global attention only for the very first token. Finally, we make sure that no loss is computed on padded tokens by setting their index to `-100`.
"""

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["passage"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch["summary1"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


#train_dataset = train_dataset.select(range(2000))
val_dataset = val_dataset.select(range(8))

"""Great, having defined the mapping function, let's preprocess the training data"""

cache_file = os.path.join(output_dir, "korean_text_summary.cache")

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=train_dataset.column_names,
    cache_file_name=cache_file, load_from_cache_file=True
)

"""and validation data"""

val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=val_dataset.column_names,
)

"""Finally, the datasets should be converted into the PyTorch format as follows."""

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

"""Alright, we're almost ready to start training. Let's load the model via the `AutoModelForSeq2SeqLM` class."""

from transformers import AutoModelForSeq2SeqLM

"""We've decided to stick to the smaller model `"allenai/led-base-16384"` for the sake of this notebook. In addition, we directly enable gradient checkpointing and disable the caching mechanism to save memory."""

led = AutoModelForSeq2SeqLM.from_pretrained(model_path, gradient_checkpointing=True, use_cache=True)
#led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)

"""During training, we want to evaluate the model on Rouge, the most common metric used in summarization, to make sure the model is indeed improving during training. For this, we set fitting generation parameters. We'll use beam search with a small beam of just 2 to save memory. Also, we force the model to generate at least 100 tokens, but no more than 512. In addition, some other generation parameters are set that have been found helpful for generation. For more information on those parameters, please take a look at the [docs](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate)."""

# set generate hyperparameters
led.config.num_beams = 2
led.config.max_length = 1024
led.config.min_length = 100
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3

"""Next, we also have to define the function the will compute the `"rouge"` score during evalution.

Let's load the `"rouge"` metric from 🤗datasets and define the `compute_metrics(...)` function.
"""

rouge = load_metric("rouge")

"""The compute metrics function expects the generation output, called `pred.predictions` as well as the gold label, called `pred.label_ids`.

Those tokens are decoded and consequently, the rouge score can be computed.
"""

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid
    
    print(f'\npred_str={pred_str}')
    print(f'\nlabel_str={label_str}')

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

"""Now, we're ready to start training. Let's import the `Seq2SeqTrainer` and `Seq2SeqTrainingArguments`."""

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

"""In contrast to the usual `Trainer`, the `Seq2SeqTrainer` makes it possible to use the `generate()` function during evaluation. This should be enabled with `predict_with_generate=True`. Because our GPU RAM is limited, we make use of gradient accumulation by setting `gradient_accumulation_steps=4` to have an effective `batch_size` of 2 * 4 = 8.

Other training arguments can be read upon in the [docs](https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainingarguments#transformers.TrainingArguments).
"""

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    fp16_backend="apex",
    output_dir=output_dir,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=5,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
)

"""The training arguments, along with the model, tokenizer, datasets and the `compute_metrics` function can then be passed to the `Seq2SeqTrainer`"""

trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

"""and we can start training. This will take about ~35min."""

trainer.train()

"""This completes the fine-tuning tutorial for LED. This training script with some small changes was used to train [this](https://huggingface.co/patrickvonplaten/led-large-16384-pubmed) checkpoint, called `" patrickvonplaten/led-large-16384-pubmed"` on a single GPU for ca. 3 days. Evaluating `" patrickvonplaten/led-large-16384-pubmed"` on Pubmed's test data gives a Rouge-2 score of **19.33** which is around 1 Rouge-2 point below SOTA performance on Pubmed.

In the Appendix below, the condensed training and evaluation scripts that were used locally to finetune `" patrickvonplaten/led-large-16384-pubmed"` are attached.

# **Appendix**

## Training
"""
exit(0)

#!/usr/bin/env python3
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# load rouge
rouge = load_metric("rouge")

# load pubmed
pubmed_train = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="train")
pubmed_val = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="validation[:10%]")

# comment out following lines for a test run
# pubmed_train = pubmed_train.select(range(32))
# pubmed_val = pubmed_val.select(range(32))

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")


# max encoder length is 8192 for PubMed
encoder_max_length = 8192
decoder_max_length = 512
batch_size = 2


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


# map train data
pubmed_train = pubmed_train.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["article", "abstract", "section_names"],
)

# map val data
pubmed_val = pubmed_val.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["article", "abstract", "section_names"],
)

# set Python list to PyTorch tensor
pubmed_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
pubmed_val.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    fp16_backend="apex",
    output_dir="./",
    logging_steps=250,
    eval_steps=5000,
    save_steps=500,
    warmup_steps=1500,
    save_total_limit=2,
    gradient_accumulation_steps=4,
)


# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# load model + enable gradient checkpointing & disable cache for checkpointing
led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384", gradient_checkpointing=True, use_cache=False)

# set generate hyperparameters
led.config.num_beams = 4
led.config.max_length = 512
led.config.min_length = 100
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=pubmed_train,
    eval_dataset=pubmed_val,
)

# start training
trainer.train()

"""## Evaluation"""

import torch

from datasets import load_dataset, load_metric
from transformers import LEDTokenizer, LEDForConditionalGeneration

# load pubmed
pubmed_test = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="test")

# load tokenizer
tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed").to("cuda").half()


def generate_answer(batch):
  inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return batch


result = pubmed_test.map(generate_answer, batched=True, batch_size=4)

# load rouge
rouge = load_metric("rouge")

print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"], rouge_types=["rouge2"])["rouge2"].mid)