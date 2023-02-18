import mii 

generator = mii.mii_query_handle("lcw_deployment")
result = generator.query({"query": ["오로라는 왜 생기는 거냐?"]}, do_sample=True, max_new_tokens=120)
print(f"{result}\n{result.response}")