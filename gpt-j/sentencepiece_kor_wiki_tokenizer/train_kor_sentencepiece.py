import os
import sentencepiece as spm
from datasets import load_dataset

data_server = os.environ['AI_DATA_SERVER']

wiki = load_dataset("lcw99/wikipedia-korean-20221001", split="train")
# for l in wiki['text']:
#     print(l) 

spm.SentencePieceTrainer.train(sentence_iterator=iter(wiki['text']), model_prefix='wiki_kor', vocab_size=32000, user_defined_symbols=['</s>', '<pad>'])