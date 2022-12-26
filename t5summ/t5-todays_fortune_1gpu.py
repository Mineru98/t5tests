
import transformers
from datasets import load_dataset, load_metric

import nltk
nltk.download('punkt')
import string, random
from transformers import AutoTokenizer, T5TokenizerFast
import evaluate

#model_checkpoint = "google/mt5-base"
#model_checkpoint = "paust/pko-t5-small"
#model_checkpoint = "paust/pko-t5-base"
model_checkpoint = "paust/pko-t5-large"

model_name = "t5-large-korean-todays-fortune-sinbiun"

model_dir = f"./Models/{model_name}"

medium_datasets = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/saju_data/sinbiun_data_by_section.json")

print(medium_datasets)

"""## Dataset train/validation/test split"""

# datasets_train_test = medium_datasets["train"].train_test_split(test_size=3000)
# medium_datasets["train"] = datasets_train_test["test"]

datasets_train_test = medium_datasets["train"].train_test_split(test_size=100)
datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=100)

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
medium_datasets["train"] = medium_datasets["train"].shuffle()
#medium_datasets["train"] = medium_datasets["train"].shuffle().select(range(10000))
medium_datasets["validation"] = medium_datasets["validation"].shuffle().select(range(64))
medium_datasets["test"] = medium_datasets["test"].shuffle().select(range(64))

print(medium_datasets)

"""## Data preprocessing"""



#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
#tokenizer = T5TokenizerFast.from_pretrained(model_dir, local_files_only=True)

prefix = "todaysfortune: "
#prefix = "xxxfff: "

max_input_length = 256
max_target_length = 1024

tokenizer.model_max_length = max_target_length

def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_titles)
  return text_cleaned

def preprocess_data(examples):
  texts_cleaned = [clean_text(text) for text in examples["source"]]
  #print(texts_cleaned)
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["overall"], max_length=max_target_length, truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

#medium_datasets_cleaned = medium_datasets.filter(lambda example: example['target'].find('재물운은 ') >= 0)
medium_datasets_cleaned = medium_datasets
#cache_file = f"./cache/{model_name}_{max_input_length}.cache"
tokenized_dataset_train = medium_datasets_cleaned["train"].map(preprocess_data, batched=True)
tokenized_dataset_val = medium_datasets_cleaned["validation"].map(preprocess_data, batched=True)

"""## Fine-tune T5"""

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration

#!rm -r {model_dir}

batch_size = 2
args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #auto_find_batch_size=True,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=30,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    report_to="tensorboard",
    gradient_accumulation_steps=8
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

import numpy as np

metric = load_metric("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
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
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    results_bleu = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    i = random.randrange(len(decoded_preds))
    print('source=', decoded_labels[i])
    print('predicion=', decoded_preds[i])
    
    result["bleu"] = results_bleu['bleu'] 
    result["length_ratio"] = results_bleu['length_ratio'] 

    return {k: round(v, 4) for k, v in result.items()}

# Function that returns an untrained model to be trained
def model_init():
    #return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    #return MT5ForConditionalGeneration.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    model.config.max_length = max_target_length
    model.config.__dict__["_name_or_path"] = model_name
    return model
     
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Commented out IPython magic to ensure Python compatibility.
# Start TensorBoard before training to monitor it in progress
# %load_ext tensorboard
# %tensorboard --logdir '{model_dir}'/runs

trainer.train()

trainer.save_model()

"""## Load the model from GDrive"""

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512

text = """
한국 경제가 심각한 위기 상황에 접어들고 있다. 급격한 금리인상으로 가계부채 부담이 늘면서 부동산 가격이 빠르게 떨어진다. 더 심각한 것은 향후 금리다. 내년에도 미국 금리에 맞춰서 지금보다 0.5% 이상 높은 금리가 1년간 지속될 것으로 보인다. 가계 부담을 생각하면 금리인상을 멈춰야 하지만 그렇게 되면 환율 위험이 있기 때문에 금리인상은 불가피하다.
"""

inputs = [prefix + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print(predicted_title)
# Session State and Callbacks in Streamlit

text = """
The special master appointed to review documents federal agents seized at Donald Trump’s Florida estate has given the former president until next Friday to back up his allegation that FBI planted evidence in the search on Aug. 8.

Following the FBI search of his Mar-a-Lago resort in Palm Beach, Trump and his lawyers have publicly insinuated on multiple occasions without providing evidence that agents had planted evidence during the search. “Planting information anyone?” Trump wrote on his Truth Social platform Aug. 12.

In an filing Thursday, Senior U.S. District Judge Raymond J. Dearie of New York, the court-appointed special master, ordered the government to turn over copies of all non-classified items seized in the case to Trump's lawyers by Monday.

He then ordered Trump's team to submit a "declaration or affidavit" of any items in the inventory that were removed from Mar-a-Lago that the "Plaintiff asserts were not seized from the Premises," meaning items that were put there by someone else.

Dearie also asked Trump's lawyers to identify any items that were seized by agents but not listed in the inventory. "This submission shall be Plaintiff’s final opportunity to raise any factual dispute as to the completeness and accuracy of the Detailed Property Inventory," the judge wrote.

Both sides were ordered to appear for a status conference in the case on Oct. 6.



"""

inputs = [prefix + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print(predicted_title)
# Conversational AI: The Future of Customer Service

