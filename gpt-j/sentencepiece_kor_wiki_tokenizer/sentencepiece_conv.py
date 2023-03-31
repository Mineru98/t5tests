import sentencepiece_model_pb2 as model

m_org = model.ModelProto()
m_org.ParseFromString(open('./StockModels/llama-7B-origianal/tokenizer.model', 'rb').read())

m_kor = model.ModelProto()
m_kor.ParseFromString(open('./StockModels/sentencepiece_wiki_kor_tokenizer/tokenizer.model', 'rb').read())

set_org = set()
for p in m_org.pieces:
    set_org.add(str(p))
count = 0
print(f"{len(m_org.pieces)=}")
for k_piece in m_kor.pieces:
    print(count)
    count += 1
    if str(k_piece) not in set_org:
        m_org.pieces.append(k_piece)                     
print(f"{len(m_org.pieces)=}")
with open('./StockModels/llama_kor_entended_tokenizer/tokenizer.model', 'wb') as f:
    f.write(m_org.SerializeToString())