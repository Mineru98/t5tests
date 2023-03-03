from rasa.core.agent import Agent
import asyncio
import glob
import os

def find_latest_file(path):
    list_of_files = glob.glob(f'{path}/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def print_entities(text):
    print(text)
    r = asyncio.run(agent.parse_message(message_data=text))
    print(r['intent']['name'], r['intent']['confidence'])
    for e in r['entities']:
        print('entity', e['entity'], e['value'], e['confidence_entity'])
    print("")
    
while True:
    file = find_latest_file("./rasa/models")
    print(f'\n\n\n\n\nloading...{file}')
    agent = Agent.load(model_path=file)

    print_entities('저알')
    print_entities('개나리 시 써봐')
    print_entities('이순신으로 삼행시 써봐')
    print_entities('안녕')
    print_entities('어떻게')
    print_entities('이순신에 대해서 블로그를 써봐')
    print_entities('북한의 인권 상황에 대해서 기사를 써봐')
    print_entities('계란 파전 만드는 법 알려줘')
    print_entities('꽃피는 봄을 소재로 시 한수 써줘')
    print_entities('오로라를 잘 볼 수 있는 곳은')
    print_entities('이순신에 대해서 설명해봐')
    print_entities('직장을 옮기고 싶어')
    print_entities('1972년 음력 11월 15일 밤 10시경 출생')
    print_entities('지금 부터 영어로')
    print_entities('지금 부터 한글로')
    print_entities('speak in english')
    
    input('waiting...')