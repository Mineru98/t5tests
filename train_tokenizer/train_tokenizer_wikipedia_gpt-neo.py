from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets

old_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
ds_wiki = load_dataset("lcw99/wikipedia-korean-20221001")
ds_namu = load_dataset("heegyu/namuwiki-extracted")
medium_datasets = concatenate_datasets([ds_wiki['train'], ds_namu['train']])

print(medium_datasets)

def get_training_corpus():
    dataset = medium_datasets
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]

example = """
제임스 얼 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.\n\n생애\n\n어린 시절 \n지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.\n\n조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다. 그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.\n\n정계 입문 \n1962년 조지아주 상원 의원 선거에서 낙선하나 그 선거가 부정선거 였음을 입증하게 되어 당선되고, 1966년 조지아 주지사 선거에 낙선하지만, 1970년 조지아 주지사를 역임했다. 대통령이 되기 전 조지아주 상원의원을 두번 연임했으며, 1971년부터 1975년까지 조지아 지사로 근무했다. 조지아 주지사로 지내면서, 미국에 사는 흑인 등용법을 내세웠다.\n\n대통령 재임 \n\n1976년 미합중국 제39대 대통령 선거에 민주당 후보로 출마하여 도덕주의 정책으로 내세워서, 많은 지지를 받고 제럴드 포드 대통령을 누르고 당선되었다.\n\n카터 대통령은 에너지 개발을 촉구했으나 공화당의 반대로 무산되었다.\n\n외교 정책 \n카터는 이집트와 이스라엘을 조정하여 캠프 데이비드에서 안와르 사다트 대통령과 메나헴 베긴 수상과 함께 중동 평화를 위한 캠프데이비드 협정을 체결했다. 이것은 공화당과 미국의 유대인 단체의 반발을 일으켰다.
"""

tokens = old_tokenizer.tokenize(example)
len_old = len(tokens)
print(len_old, tokens)

training_corpus = get_training_corpus()
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, len(old_tokenizer))

tokens = tokenizer.tokenize(example)
len_new = len(tokens)
print(len_new, tokens)
print("rate=", len_new/len_old)

tokenizer_name = "tokenizer-wiki-plus-namu-gpt-neo-ko"
tokenizer.save_pretrained(f"./{tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(f"./{tokenizer_name}")

tokens = tokenizer.tokenize(example)
print(len(tokens), tokens)

