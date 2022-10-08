from transformers import AutoTokenizer, AutoModelForCausalLM
import re

model_name = "EleutherAI/gpt-j-6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_bert = AutoTokenizer.from_pretrained("kykim/bert-kor-base")

tokenizer.save_pretrained("../train_tokenizer/tokenizer-gpt-j-6B-org")

vocabulary = list(tokenizer.get_vocab().keys())
print(vocabulary[:100])

vocabulary_bert = list(tokenizer_bert.get_vocab().keys())
print(vocabulary_bert[:100])

new_tokens = []
for word in vocabulary_bert:
    if word not in vocabulary:
        if not re.match('^#+[a-zA-Z0-9_]+$', word): 
            new_tokens.append(word)
        else:
            print(word)

print("adding new tokens count=", len(new_tokens))
tokenizer.add_tokens(new_tokens)
print("done, adding")

tokenizer.save_pretrained("../train_tokenizer/tokenizer-gpt-j-plus-ko")
print("done, saving")