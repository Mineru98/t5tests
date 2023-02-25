from rasa.core.agent import Agent
import asyncio

agent = Agent.load(model_path='./rasa/models/model.tar.gz')

result = asyncio.run(agent.parse_message(message_data='이순신에 대해서 말해봐.'))
print(result)
result = asyncio.run(agent.parse_message(message_data='이순신에 대해서 블로그를 써봐.'))
print(result)
result = asyncio.run(agent.parse_message(message_data='북한의 인권 상황에 대해서 기사를 써봐.'))
print(result)
result = asyncio.run(agent.parse_message(message_data='파전 만드는 법 알려줘.'))
print(result)