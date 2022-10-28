from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("../train_tokenizer/tokenizer-wiki-plus-namu-gpt-neo-ko")

gpt = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")

# tokenizer_len = len(tokenizer)
# print("\n\n\n=====\ntokenizer_len=", tokenizer_len)
# gpt.resize_token_embeddings(tokenizer_len)

gpt.save_pretrained("./StockModels/gpt-neo-1.3B")