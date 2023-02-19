import mii 
import asyncio, time

response = []
generator = mii.mii_query_handle("lcw_deployment")
def main(text):
    result = generator.query_non_block({"query": [f"{text}"]}, do_sample=True, max_new_tokens=120)
    return result

id = main("하늘은 왜 파란가?")
id2 = main("오로라는 왜 생기나?")

while True:
    result = generator.get_pending_task_result(id2)
    if result is not None:
        print(result.response[0])
        break