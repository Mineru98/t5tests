from rasa.core.agent import Agent
import asyncio

agent = Agent.load(model_path='./rasa/models/model.tar.gz')

result = asyncio.run(agent.parse_message(message_data='안녕?'))
print(result)
print(result['intent']['name'])
result = asyncio.run(agent.parse_message(message_data='이순신에 대해서 블로그를 써봐.'))
print(result)
print(result['intent']['name'])
result = asyncio.run(agent.parse_message(message_data='북한의 인권 상황에 대해서 기사를 써봐.'))
print(result)
print(result['intent']['name'])
result = asyncio.run(agent.parse_message(message_data='계란 파전 만드는 법 알려줘.'))
print(result)
print(result['intent']['name'])
