from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration
from transformers import AutoTokenizer, T5TokenizerFast
import nltk, os, glob
nltk.download('punkt')

model_name = "t5-large-korean-paraphrase"
model_dir = f"./Models/{model_name}"
#latest_model_dir = max(glob.glob(os.path.join(model_dir, 'checkpoint-*/')), key=os.path.getmtime)
latest_model_dir = "/home/chang/AI/llm/t5tests/t5summ/Models/t5-large-korean-paraphrase/checkpoint-269000"
print(f'loading model={latest_model_dir}')
tokenizer = AutoTokenizer.from_pretrained(latest_model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(latest_model_dir)

max_input_length = 512

text = []
text.append("""
실제 조봉행 체포 작전에 협조했던 '협력자 K씨'도 홍어 사업이 아니라 수리남에 선박용 특수용접봉을 파는 사업을 하러 수리남에 갔었다.

""")

text.append("""
36년생 운기가 살아나니 망설이지 말고 나서라 48년생 사람 관계도 정리와 폐기가 필요 60년생 정신세계를 가꾸지 않은 삶은 척박하다 72년생 오랫동안 바라고 소망한 것을 이룰 듯. 84년생 장점과 단점 모두 자산. 96년생 주위 범띠가 귀인.

""")

text.append("""
“아니, 원래 직업이 이건 아닐 거 아냐? 원래 뭐 했어? 낮엔 뭐 해? 집은 어디야? 명찰엔 왜 홍금보라고 적었어?”
""")

for i in range(3):
    print(text[i])
    inputs = ["paraphrase: " + text[i]]
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=100, max_length=200, num_return_sequences=3)
    for decoded_output in tokenizer.batch_decode(output, skip_special_tokens=True):
        predicted_title = nltk.sent_tokenize(decoded_output.strip())
        print(">>", predicted_title)

while True:
    print("\n")
    text = input("Input: ")
    inputs = ["paraphrase: " + text]
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=100, max_length=200, num_return_sequences=3)
    for decoded_output in tokenizer.batch_decode(output, skip_special_tokens=True):
        predicted_title = nltk.sent_tokenize(decoded_output.strip())
        print(">>", predicted_title)
