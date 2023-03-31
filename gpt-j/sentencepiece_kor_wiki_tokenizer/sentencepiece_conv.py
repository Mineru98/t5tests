import sentencepiece_model_pb2 as model
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

train = True
if train:
    m_org = model.ModelProto()
    m_org.ParseFromString(open('./StockModels/llama-7B-origianal/tokenizer.model', 'rb').read())

    m_kor = model.ModelProto()
    m_kor.ParseFromString(open('./StockModels/sentencepiece_wiki_kor_tokenizer/tokenizer.model', 'rb').read())

    set_org = set()
    for p in m_org.pieces:
        set_org.add(p.piece)
    count = 0
    print(f"{len(m_org.pieces)=}")
    for k_piece in m_kor.pieces:
        print(count)
        count += 1
        if str(k_piece.piece) not in set_org:
            m_org.pieces.append(k_piece)                     
    print(f"{len(m_org.pieces)=}")
    with open('./StockModels/llama_kor_entended_tokenizer/tokenizer.model', 'wb') as f:
        f.write(m_org.SerializeToString())
    
target_tokenizer = AutoTokenizer.from_pretrained('./StockModels/llama_kor_entended_tokenizer')
target_tokenizer.pad_token = target_tokenizer.eos_token
target_tokenizer.pad_token_id = target_tokenizer.eos_token_id

target_tokenizer.save_pretrained('./StockModels/llama_kor_entended_tokenizer_special_tokens')