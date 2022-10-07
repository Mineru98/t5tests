from transformers import GPTJForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../train_tokenizer/tokenizer-gpt-j-plus-ko")

gpt = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", device_map='auto', load_in_8bit=True)

tokenizer_len = len(tokenizer)
print("\n\n\n=====\ntokenizer_len=", tokenizer_len)
gpt.resize_token_embeddings(tokenizer_len)

gpt.save_pretrained("./gpt-j-6B-8bit-ko-voc")