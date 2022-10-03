from transformers import AutoTokenizer
from datasets import load_dataset, load_metric

old_tokenizer = AutoTokenizer.from_pretrained("./tokenizer_SNS_Korean")
medium_datasets = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/한국어 SNS/korean_sns_training_gpt2_v2.json")
print(medium_datasets)
datasets_train_test = medium_datasets["train"].train_test_split(test_size=2000000)
print(datasets_train_test)

def get_training_corpus():
    dataset = datasets_train_test["test"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        ss = []
        for s in samples["sample"]:
            ss.append(s.replace("\t", " ").replace("\n", " "))
        samples["sample"] = ss
        yield samples["sample"]

example = """
밑에가 훨 싸네요 저기도 가깝지 않아요!! 하루 17만원 인거죠? 밑에꺼 해욧!! 응응 두번째보다는 멀지망 가까운편 오키
"""

tokens = old_tokenizer.tokenize(example)
print(tokens)

training_corpus = get_training_corpus()
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 51200)

tokens = tokenizer.tokenize(example)

tokenizer.save_pretrained("./tokenizer_SNS_Korean2")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_SNS_Korean2")

tokens = tokenizer.tokenize(example)
print(tokens)
