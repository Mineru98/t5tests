from transformers import AutoTokenizer, AutoModelForCausalLM
import re

model_name = "EleutherAI/gpt-j-6B"

tokenizer = AutoTokenizer.from_pretrained("../train_tokenizer/tokenizer-gpt-j-6B-org")
tokenizer_ko_wiki = AutoTokenizer.from_pretrained("../train_tokenizer/tokenizer_wikipedia_gpt_j")

tok_len = len(tokenizer)
print("len(tokenizer)=", tok_len)
print("len(tokenizer_ko_wiki)=", len(tokenizer_ko_wiki))

vocabulary = list(tokenizer.get_vocab().keys())
print(vocabulary[:100])

vocabulary_bert = list(tokenizer_ko_wiki.get_vocab().keys())
print(vocabulary_bert[:100])

new_tokens = []
for word in vocabulary_bert:
    if word not in vocabulary:
        if not re.match('^#+[a-zA-Z0-9_]+$', word): 
            new_tokens.append(word)
        else:
            print(word)
len_new_tokens = len(new_tokens)
remove_n = tok_len + len_new_tokens - 91238
new_tokens = new_tokens[:len_new_tokens - remove_n]
print("adding new tokens count=", len(new_tokens))
tokenizer.add_tokens(new_tokens)
print("done, adding")
print("len(tokenizer)=", len(tokenizer))

tokenizer.save_pretrained("../train_tokenizer/tokenizer-gpt-j-plus-ko-v2")
print("done, saving")