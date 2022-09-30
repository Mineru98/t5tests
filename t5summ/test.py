from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration
from transformers import AutoTokenizer, T5TokenizerFast
import nltk
nltk.download('punkt')

model_name = "t5-large-korean-text-summary"
model_dir = f"./Models/{model_name}/checkpoint-26000"
#model_dir = "lcw99/t5-base-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512 + 256

text1 = """
주인공 강인구(하정우)는 ‘수리남에서 홍어가 많이 나는데 다 갖다버린다’는 친구 
박응수(현봉식)의 얘기를 듣고 수리남산 홍어를 한국에 수출하기 위해 수리남으로 간다. 
국립수산과학원 측은 “실제로 남대서양에 홍어가 많이 살고 아르헨티나를 비롯한 남미 국가에서 홍어가 많이 잡힌다”며 
“수리남 연안에도 홍어가 많이 서식할 것”이라고 설명했다.

그러나 관세청에 따르면 한국에 수리남산 홍어가 수입된 적은 없다. 
일각에선 “돈을 벌기 위해 수리남산 홍어를 구하러 간 설정은 개연성이 떨어진다”는 지적도 한다. 
드라마 배경이 된 2008~2010년에는 이미 국내에 아르헨티나, 칠레, 미국 등 아메리카산 홍어가 수입되고 있었기 때문이다. 
실제 조봉행 체포 작전에 협조했던 ‘협력자 K씨’도 홍어 사업이 아니라 수리남에 선박용 특수용접봉을 파는 사업을 하러 수리남에 갔었다.

"""

text2 = """
“야간 알바 구하신다고 해서 찾아왔습니다.”
순간 자동으로 입꼬리가 실룩거렸다. 마스크가 표정의 상당 부분을 감춰준다는 게 다행이었다. 선숙은 빠르게 사내를 스캔했다. 커다란 눈과 처진 눈썹이 어딘가 초식동물을 연상케 했고, 겨자색인지 똥색인지 모를 목 늘어난 티셔츠에 헝클어진 곱슬머리는 전체적으로 구질구질해 보이는 인상이었다.
“알바 지원하러 왔다며 화장지는 왜 사는 거예요?”
“그게, 저희 어머니가 어디 아는 가게 가면 꼭 팔아줘야 한다고 하셨거든요. 마침 집에 휴지도 떨어졌고 해서요. 아하하.”
뭐지? 이 과한 예의는? 부담스러운 면이 없지 않았으나 사람 좋게 웃는 모습에 다소 마음이 놓이기는 했다. 무엇보다 야간 알바 자원이었다. 깐깐하게 굴기보다는 일단 뽑고 볼 일이었다.

"""

text3 = """
장사가 안 돼도, 코로나에 세상이 엉망이어도, 이 녀석은 명찰에 ‘홍금보’라고 써놓고 헤실헤실 웃고만 있다. 참으로 부러운 재능이다. 한마디로 멘탈 금수저다. 나이는 마흔 넘은 게 분명한데 편의점 야간 알바나 하는 형편에 뭐가 그리도 느긋한지.
“어이, 홍금보. 자네 정체가 뭐야?”
계산을 마치고 카드를 건네는 녀석에게 물었다.
“저요? 편의점 야간 알바죠.”
“아니, 원래 직업이 이건 아닐 거 아냐? 원래 뭐 했어? 낮엔 뭐 해? 집은 어디야? 명찰엔 왜 홍금보라고 적었어?”
“음…… 원래부터 전 알바하며 살았어요. 예전엔 노가다도 좀 했구요. 낮엔 잡니다. 밤에 일하면 낮에 수면의 질이 안 좋아서 오래 자줘야 해요. 집은 저기 남대문시장 위 남창동 살구요…… 또 뭐 물으셨죠? 아, 홍금보는 어릴 적부터 별명입니다. 제 본명이 근배거든요. 황근배. 아하하.”
"""

print("")
inputs = ["summarize: " + text1]
inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=128, num_return_sequences=3)
for decoded_output in tokenizer.batch_decode(output, skip_special_tokens=True):
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
    print(">>", predicted_title)

print("")
inputs = ["summarize: " + text2]
inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=128, num_return_sequences=3)
for decoded_output in tokenizer.batch_decode(output, skip_special_tokens=True):
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
    print(">>", predicted_title)

print("")
inputs = ["summarize: " + text3]
inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=128, num_return_sequences=3)
for decoded_output in tokenizer.batch_decode(output, skip_special_tokens=True):
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
    print(">>", predicted_title)