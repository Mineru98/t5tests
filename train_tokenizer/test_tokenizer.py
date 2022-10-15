from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
tokenizer = AutoTokenizer.from_pretrained("lcw99/tokenizer-wiki-plus-namu-gpt-j-ko")
example = """
경상북도 안동군[33] 출생으로 안동에서 초등학교를 졸업 후 경기도 성남시로 이주하여 소년공 생활을 했다. 검정고시를 통해 중졸·고졸 학력을 취득한 뒤 중앙대학교 법과대학에 진학했고, 대학을 졸업하고 사법시험에 합격 후 법조계의 길로 들어섰다. 이후 경기도 성남시 일대에서 인권변호사 겸 시민사회운동가로 활동했다.

2010년 지방선거에서 성남시장에 처음 당선되었고, 2014년 지방선거에서 성남시장 재선에 성공하였다. 2017년 더불어민주당 제19대 대통령 후보 경선에 참여했지만 3위로 낙선했다. 이후 2018년 지방선거에서 경기도지사에 당선되었다.[34]

2021년 7월 1일, 제20대 대통령 선거 출마를 공식 선언하였고, 더불어민주당 제20대 대통령 후보 경선에 참여하여 2021년 10월 10일 더불어민주당 대통령 후보로 선출되었다. 2022년 3월 9일 치러진 제20대 대통령 선거에서 국민의힘 윤석열 후보에게 0.73%p[35] 차이로 밀려 2위로 낙선하였다.

이후 2022년 6월 보궐선거에서 인천광역시 계양구 을 지역구 국회의원에 출마하여 당선되었으며, 8월 28일 열린 더불어민주당 2022년 전당대회에서 77.77%의 압도적인 득표율로 제6대 당대표에 당선되었다.

"""
# example =  """
# A dozen Russian missile strikes on apartment buildings and other targets in Zaporizhzhia killed at least 12 civilians and injured dozens more. The southeastern city, about 50 kilometers from the namesake nuclear power plant, has been under escalating attacks but Sunday’s strikes were the deadliest so far. Ukrainian officials stepped up their calls for air missile defense systems from Western allies. 
# """
len_source = len(example)
print(len_source)
tokens = tokenizer.tokenize(example) 
len_token = len(tokens)
print(len_token, tokens)

print("rate=", len_token / len_source)

 