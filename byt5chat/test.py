from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration
from transformers import AutoTokenizer, T5TokenizerFast
import nltk
nltk.download('punkt')

model_name = "byt5-base-korean-chit-chat"
model_dir = f"./Models/{model_name}/checkpoint-1000"
#model_dir = "paust/pko-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 1024

text = """
A: 상태가 이상해 B: 
"""

inputs = [text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=100, max_length=500, num_return_sequences=3)
for i in range(3):
    print(output[i])
    decoded_output = tokenizer.decode(output[i], skip_special_tokens=True)
    predicted_title = nltk.sent_tokenize(decoded_output)
    print(decoded_output)
    print(predicted_title)