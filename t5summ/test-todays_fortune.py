from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration
from transformers import AutoTokenizer, T5TokenizerFast
import nltk, os, glob
import pandas as pd

nltk.download('punkt')

def get_todays_fortune(gender, birthday, target_date):
    url = f'https://fortune.stargio.co.kr:28080/todayLuck/woonse?gender={gender}&saju={birthday}&loveDate={target_date}'
    tables = pd.read_html(url, encoding='utf-8') 
    t = tables[0] 
    #print(t)
    c = 0
    s = f"사주를 보자면 시주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 1
    s += f"일주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 2
    s += f"월주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 3
    s += f"년주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이다."       
    #print(s)
    saju = s 

    t = tables[1] 
    #print(t)
    c = 0
    s = f"들어온 날을 보자면 일주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 1
    s += f"월주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 2
    s += f"년주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이다."       
    #print(s)
    target_date_samju = s

    t = tables[2] 
    #print(t)
    s = t[0][1]
    #print(s)
    fortune = s
    
    return saju, target_date_samju, fortune


model_name = "t5-base-korean-todays-fortune-sinbiun"
model_dir = f"./Models/{model_name}"
latest_model_dir = max(glob.glob(os.path.join(model_dir, 'checkpoint-*/')), key=os.path.getmtime)
print(f'loading model={latest_model_dir}')
tokenizer = AutoTokenizer.from_pretrained(latest_model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(latest_model_dir)

max_input_length = 512

text = []
saju, target_date_samju, fortune = get_todays_fortune('female', '199010121111', '20201011')
text.append(f'{saju} {target_date_samju}')

for i in range(len(text)):
    print(text[i])
    inputs = ["todaysfortune: " + text[i]]
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=100, max_length=1024, num_return_sequences=1)
    for decoded_output in tokenizer.batch_decode(output, skip_special_tokens=True):
        predicted_title = nltk.sent_tokenize(decoded_output.strip())
        print(">>", predicted_title)

while True:
    print("\n")
    sex = input("sex: ")
    birthday = input("birthday: ")
    target_date = input("target date: ")
    saju, target_date_samju, fortune = get_todays_fortune(sex, birthday, target_date)
    text = f'{saju} {target_date_samju}'
    print(text)
    
    inputs = ["todaysfortune: " + text]
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=100, max_length=1024, num_return_sequences=1)
    for decoded_output in tokenizer.batch_decode(output, skip_special_tokens=True):
        predicted_title = nltk.sent_tokenize(decoded_output.strip())
        print(">>", predicted_title)
