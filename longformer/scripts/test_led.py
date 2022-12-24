import torch

from datasets import load_dataset, load_metric
from transformers import LEDTokenizer, LEDForConditionalGeneration, AutoTokenizer

# load pubmed
# pubmed_test = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="test")

# load tokenizer
model = './led_model_ko_summ'
tokenizer = AutoTokenizer.from_pretrained(model)
model = LEDForConditionalGeneration.from_pretrained(model).to("cuda")
model.save_pretrained("./StockModels/led_ko_summ")
tokenizer.save_pretrained("./StockModels/led_ko_summ")

exit(0)
def generate_1(text):
  inputs_dict = tokenizer(text, padding="max_length", max_length=4096, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, max_length=1024)
  return tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
      
def generate_answer(batch):
  result = generate_1(batch["article"])
  batch["predicted_abstract"] = result
  return batch

sample = """
1973년 삼환기업이 사우디아라비아의 고속도로 건설공사를 수주해 중동 시장에 첫 진출했다. 사우디는 "40일 만에 공사를 끝내 달라"는 조건을 달았다. 근로자 수천 명이 밤에 횃불을 밝히고 도로 건설에 매진했다. 한밤중 횃불에 사우디 국왕이 "난동이 났느냐"고 놀랐다가 사정을 듣고서는 "저렇게 부지런하고 성실한 사람들에겐 공사를 더 많이 맡기라"고 했다.

'사우디 횃불 신화'를 바탕으로 1976년엔 현대건설이 사우디 주베일 항만 공사를 수주했다. 공사 금액이 9억4천만 달러, 당시 우리나라 한 해 예산의 25%에 달하는 대규모 공사였다. 1978년엔 무려 9만 명 이상이 중동 건설 현장에 진출했다.

중동 건설 현장에서 보여준 한국 근로자들의 열정에 중동 국가들은 경탄했다. 강인한 정신력과 근면함은 한국 근로자의 트레이드마크가 됐다. 근로자들이 보여준 불굴의 모습은 한국과 한국인에 대한 좋은 인상을 심어줘 우리 기업들이 세계 각지에 진출하는 데 긍정적 작용을 했다.

무함마드 빈 살만 사우디 왕세자 겸 총리의 한국 방문에서 한국과 사우디가 26건의 투자 계약 및 양해각서(MOU)를 체결했다. 300억 달러(40조2천억 원)에 달한다. 670조 원을 들여 '네옴시티' 건설에 나선 빈 살만 왕세자가 한국에 손을 내민 것은 근면과 불굴의 정신력 등 한국인이 보여준 DNA를 신뢰하기 때문이다. 사우디에서 우리 근로자들이 보여준 신화를 다시 한번 실현해 주길 바랄 것이다.

사우디에서 촉발한 '제2의 중동 특수'를 우리 것으로 가져오려면 우리에게서 사라진 근면과 불굴의 정신력과 같은 DNA를 되살리는 게 필수다. 오일 쇼크에 빠진 한국 경제를 위기에서 건진 1970년대 중동의 한국 근로자들이 보여준 DNA를 되찾아야 하는 것이다.

그러고 보니 근면, 자조, 협동 등 우리에게서 사라진 DNA가 많다. 가장 심각한 것은 옳고 그름을 판단하는 DNA가 없어진 것이다. 모든 사안을 네 편, 내 편에 따라 판단을 정반대로 하는 '이상한 나라'가 되고 말았다.
"""

result = generate_1(sample)
print(result)

# result = pubmed_test.map(generate_answer, batched=True, batch_size=4)

# # load rouge
# rouge = load_metric("rouge")

# print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"], rouge_types=["rouge2"])["rouge2"].mid)
