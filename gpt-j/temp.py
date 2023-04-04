tokenizer_id = "/home/chang/AI/llm/t5tests/gpt-j/Models/llama-7B-en-alpaca"

from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers import AutoTokenizer
from tokenizers import Tokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/chang/AI/llm/t5tests/gpt-j/StockModels/llama-7B-origianal")

if False:
    new_tokenizer = Tokenizer.from_file("tok.json")
else:
    new_tokenizer = convert_slow_tokenizer(tokenizer)
    new_tokenizer.save("tok.json")

strings = [
    "This is a test",
    "生活的真谛是",
    "生活的真谛是[MASK]。",
    # XXX: This one is problematic because of special tokens
    # "<s> Something something",
]

for string in strings:
    encoded = tokenizer(string)["input_ids"]
    encoded2 = new_tokenizer.encode(string).ids

    assert encoded == encoded2, f"{encoded} != {encoded2}"

    decoded = tokenizer.decode(encoded)
    decoded2 = new_tokenizer.decode(encoded2)

    assert decoded.strip() == decoded2, f"{repr(decoded)} != {repr(decoded2)}"