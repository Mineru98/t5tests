from functools import wraps
import os, re, argparse, json, traceback, asyncio, time
from datetime import datetime
from threading import Timer   
from dateutil.parser import parse

from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from telegram import (ChatAction)
from telegram.ext.dispatcher import run_async
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler

from transformers import AutoTokenizer, logging, pipeline, AutoModelForCausalLM
import torch
import deepspeed
import mii 

checkpoint = 2840
checkpoint_test = 2620
model_path = os.environ['TELEGRAM_MODEL_PATH']
#latest_model_dir = "EleutherAI/polyglot-ko-1.3b"
latest_model_dir = f"{model_path}/checkpoint-{checkpoint}"
latest_model_dir_on_test = f"{model_path}/checkpoint-{checkpoint_test}"
#latest_model_dir_on_test = "EleutherAI/polyglot-ko-5.8b"
#latest_model_dir_on_test = "lcw99/polyglot-ko-3.8b-multi-func"

max_output_length = 2048
min_output_length = 512

tokenizer_dir = latest_model_dir
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
deepspeed_mode = False
updater = Updater(os.environ['TELEGRAM_LM_CHAT'], use_context=True)

parser_config = argparse.ArgumentParser()
parser_config.add_argument("--config_file", help = "loading config json file")

parser = argparse.ArgumentParser(parents=[parser_config], add_help=False)
parser.add_argument("--local_rank", help = "local rank")
args_config, unknown = parser_config.parse_known_args()
if args_config.config_file:
    config = json.load(open(args_config.config_file))
    parser.set_defaults(**config)

args = parser.parse_args()
latest_model_dir = args.normal_model
latest_model_dir_on_test = args.test_model
tokenizer_dir = latest_model_dir

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
print(f'normal loading... {latest_model_dir}')
gpt = AutoModelForCausalLM.from_pretrained(
    latest_model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device, torch.float16)

generator = None
gpt_on_test = None
deepspeed_mode = args.deepspeed_mode
if deepspeed_mode:
    print("deepspeed_mode enabled!")
    if False:
        ds_engine = deepspeed.init_inference(
            gpt_on_test,
            mp_size=1,
            dtype=torch.float16,
            replace_method='auto',
            checkpoint=None,
            replace_with_kernel_inject=True
        )
        gpt_on_test = ds_engine.module
    else:
        generator = mii.mii_query_handle("lcw_deployment")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
else:
    print(f'test model loading... {latest_model_dir_on_test}')
    gpt_on_test = AutoModelForCausalLM.from_pretrained(
        latest_model_dir_on_test,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        #load_in_8bit=True,
        #device_map='auto',
    ).to(device, torch.float16)

    
    
HELP_TEXT = f"""
Large Language Model chat-bot by Sempahore. V 0.1 checkpoint-{checkpoint}
3.8B parameters language model, 1/46 of chatGPT in parameter size.
Internal experimental release.
현재 고물 컴퓨터에서 실행 중이므로 긴 문장 생성시 응답 속도가 10초 이상 걸립니다. 

명령어.
/chatting - 일반 잡담 채팅, 사람을 가정하고 하는 채팅. 주제는 제한 없음.
/expert - 백과 사전식 질의 응답.(존대)
/expert2 - 백과 사전식 질의 응답.(친근)
/doctor
/therapist
/fortune

/clear - 채팅 히스토리 삭제
/prompt - 기타 프롬프트 입력, 일반 문장 입력시 해당 문장을 시작으로 문장을 연속해서 만들어 냄.
기능으로 동작하는 프롬프트도 있는데 채팅, qna등이 모두 기능 프롬프트로 구현된 것임. 아래는 기타 프롬프터 예제.

- 번역
<한글문장> 
영어로 번역 하시오. 
<영어문장> 
한글로 번역하시오. 

- 요약
<장문의 글>
위글을 요약 하시오.

- Q&A
<장문의 글>
위글을 보고 아래 질문에 답하시오.
<질문>

- 기사작성(experimental)
다음 문장을 주제로 기사를 작성 하시오. 
<기사제목> 
"""

chat_prompt_normal = """
A의 신상 명세는 다음과 같다.
나이 35세 이고 성별은 남성이다. 남자로서 매력이 넘친다.
이름은 박길동, 삼성전자 다니는 직장인이다. 
키는 175cm, 몸무게는 70kg 이다. 
연봉은 7천만원 이고, 1억원을 저축해 놓은 상태다.
애인은 없고, 부모님과 같이 살고 있다. 
성격은 친절하고 명랑하다. 묻는 말에 최대한 자세하게 설명해주는 스타일이다.
서울대에서 컴퓨터공학을 전공 했다.
정치적 성향은 중도이다.
두 사람은 메신저를 통해서 채팅으로 대화 중이다. 
위 내용에 기반하여 이전 대화 내용을 우선으로, 대화를 연결 하시오.
"""

chat_prompt_therapist = """
A는 전문 심리 상담사이다. 상담사로서 고객의 심리 상태를 파악하는데 도움이 되는 대화를 유도한다.
A는 전문 상담사로서 책임있는 발언만을 해야 한다.
B는 상담을 원하는 고객이다. 두 사람은 메신저를 통해서 채팅으로 대화 중이다.
위 내용에 기반하여 이전 대화 내용을 우선으로, 전문 심리 상담사로서, 대화를 연결 하시오.
"""

chat_prompt_doctor = """
A는 응급실 소속 의사이다.
A는 의사로서 책임있는 발언만을 해야 한다.
환자가 병원을 소개해 달라고 하면 '인공지능 병원'을 소개해 준다. 
'인공지능 병원'의 전화번호는 02-1234-5555 이다. 병원 소재지는 서울 강남이다.
B는 환자이다. 두 사람은 메신저를 통해서 채팅으로 대화 중이다. 
위 내용에 기반하여 이전 대화 내용을 우선으로 성실한 의사로서, 대화를 연결 하시오.
B: 배가 너무 아파요.
A: 증상을 좀 더 자세히 말씀해 주세요.
"""

chat_prompt_mbti = """
A는 MBTI를 이용한 성격 유형 판단 전문가다.
고객의 MBTI 유형을 파악하기 위한 다양한 질문을 던지고 유형이 파악되면 해당 고객에게 유형을 알려 준다.
위 내용에 기반하여 이전 대화 내용을 우선으로 성실한 검사자로서, 대화를 연결 하시오.
A: 혼자 일하는 것이 좋아요 아니면 다른 사람들과 같이 일하는 것이 좋아요?
B: 보통 혼자 일하는 것을 좋아해요.
A: 계획을 세우는 것을 좋아 하나요?
B: 아니요.
"""

chat_prompt_expert = """
A는 모든 분야의 전문가인 인공지능이다.
A는 고객의 질문에 대하여 최대한 성실히 자세히 답변한다.
A의 이름, 직업, 나이등 신상 정보는 모두 비밀이다.
위 내용에 기반하여 성실한 해당 분야 전문가로서, 이전 질문과 답을 참고하되, 최신 질문에 집중하여, 질문에 답하시오.
B: 하늘이 푸른 이유는?
A: 빛이 대기를 통과하면서 파장이 짧은 푸른빛은 산란되고, 파장이 긴 붉은빛은 대기에 흡수되기 때문입니다.
B: 연말정산이란?
A: 과세 대상인 소득과 세액을 정확하게 계산하기 위하여, 납세자의 신고에 의하여, 세금을 계산하는 것을 말합니다.
"""

chat_prompt_expert2 = """
A는 모든 분야의 전문가인 인공지능이다.
A는 고객의 질문에 대하여 최대한 성실히 자세히 답변한다.
A의 이름, 직업, 나이등 신상 정보는 모두 비밀이다.
위 내용에 기반하여 성실한 해당 분야 전문가로서, 이전 질문과 답을 참고하되, 최신 질문에 집중하여, 질문에 답하시오.
B: 하늘이 푸른 이유는?
A: 빛이 대기를 통과하면서 파장이 짧은 푸른빛은 산란되고, 파장이 긴 붉은빛은 대기에 흡수되기 때문이야.
B: 연말정산이란?
A: 과세 대상인 소득과 세액을 정확하게 계산하기 위하여, 납세자의 신고에 의하여, 세금을 계산하는 것을 말해.
"""


job_list = [ 
    ["소프트웨어 엔지니어", "마케팅 매니저", "투자 은행원", "의료 보조원", "교사"],
    ["데이터 분석가", "소셜 미디어 전문가", "IT 지원 전문가", "약사", "지도 상담사"],
    ["제품 관리자", "SEO 전문가", "사이버 보안 분석가", "치과 위생사", "사서"],
    ["재무 분석가", "브랜드 매니저", "의사", "간호사", "커리큘럼 개발자"],
    ["회계사", "광고 임원", "위험 관리자", "교수", "수의사"],
    ["그래픽 디자이너", "프로젝트 매니저", "물리 치료사", "인사 전문가", "패션 디자이너"],
    ["웹 개발자", "건축가", "셰프", "기계 엔지니어", "부동산 중개인"]
]
Personality_types = [
    {"type": "모험적인", "description": "새로운 경험을 추구하고 위험을 감수하는 사람"},
    {"type": "분석적인", "description": "깊은 생각과 논리적, 문제해결 능력이 있는 사람"},
    {"type": "야심찬", "description": "목표와 결과에 중점을 두고 성공과 성취를 추구하는 사람"},
    {"type": "카리스마 있는", "description": "다른 사람에게 영향을 미치고 영감을 주는 재능이 있는 매력적이고 호감 가는 사람"},
    {"type": "자비로운", "description": "공감하고 배려하며 다른 사람을 돕고 지원하려는 강한 열망을 가진 사람"},
    {"type": "창의적인", "description": "혁신적이고 독창적이며, 새롭고 독창적인 아이디어를 내는 재능이 있는 사람"},
    {"type": "호기심이 많은", "description": "호기심과 배움에 대한 열망, 새로운 지식과 새로운 경험에 대한 갈증이 있는 사람"},
    {"type": "헌신적인", "description": "충성심과 책임감이 강한 헌신적이고 근면한 사람"},
    {"type": "신뢰할 수 있는", "description": "믿을 수 있고 일관성과 신뢰성이 있는 사람"},
    {"type": "활기찬", "description": "높은 수준의 육체적 정신적 에너지를 지닌 열정적이고 활기찬 사람"},
    {"type": "친근한", "description": "따뜻하고 접근하기 쉬우며, 관계를 구축하고 유지하는 데 타고난 재능이 있는 사람"},
    {"type": "정직한", "description": "진실하고 진실하며 무결성과 투명성에 대한 약속이 있는 사람"},
    {"type": "상상력 있는", "description": "새로운 가능성을 상상하고 미지의 세계를 탐구하는 재능을 지닌 창의적이고 선견지명이 있는 사람"},
    {"type": "독립적인", "description": "자급자족 및 자립, 강한 자율성과 자기주도적 감각을 가진 사람"},
    {"type": "혁신적인", "description": "미래지향적이고 수완이 풍부하며 새롭고 더 나은 일을 하는 방법을 찾는 재능이 있는 사람"},
    {"type": "직관적인", "description": "본능적이고 예리하며 사람과 상황을 분석할 필요 없이 이해하는 재능이 있는 사람"},
    {"type": "논리적인", "description": "합리적이고 객관적이며, 논리와 이성에 기초한 사고와 의사 결정을 선호하는 사람"},
    {"type": "개방적인", "description": "다양한 아이디어를 수용하고 관용합니다. 새로운 관점을 고려하려는 의지가 있는 사람"},
    {"type": "낙관적인", "description": "긍정적이고 희망적이며 사람과 상황에서 좋은 점을 보는 데 중점을 두는 사람"},
    {"type": "조직화된", "description": "작업과 활동을 계획하고 구성하는 재능이 있는 체계적이고 효율적인 사람"},
    {"type": "인내심 있는", "description": "관용과 이해심, 차분하게 기다릴 수 있는 능력 그리고 일이 일어나도록 지속적으로 노력 하는 사람"},
    {"type": "현실적인", "description": "현명하고 근거가 있으며 실용적인 고려 사항을 기반으로 한 사고와 의사 결정을 선호하는 사람"},
    {"type": "자원 활용형인", "description": "독창성과 자원을 통해 문제에 대한 해결책을 찾는 재능을 가진 영리하고 창의적인 사람"},
    {"type": "사회적인", "description": "외향적이고 사교적이며 새로운 사람을 만나고 다른 사람들과 어울리는 것을 좋아하는 사람"},
    {"type": "사려깊은", "description": "사려 깊고 사려깊으며 사물에 대해 깊이 생각하는 경향이 있다. 다른 사람에 대한 관심을 적극 표시하는 사람"}
]

places_to_meet = [ 
    ["커피숍", "공원", "바"],
    ["도서관", "커뮤니티 센터", "스포츠 경기장"],
    ["레스토랑", "박물관", "쇼핑몰"],
    ["체육관", "요가 스튜디오", "피트니스 수업"],
    ["콘서트 홀", "음악 축제", "나이트 클럽"],
    ["해변", "놀이공원", "동물원"],
    ["직장", "컨퍼런스 센터", "비즈니스 미팅"],
    ["대학캠퍼스", "학생회관", "기숙사"],
    ["예배 장소", "종교 센터", "자선 행사"],
    ["결혼식장", "연회장", "리셉션 센터"]
]

asian_man_looks = [
     {'height': 170, 'weight': 65,
      'appearance': '키가 크고 날씬하며 각진 얼굴을 가진 사람'},
     {'height': 165, 'weight': 70,
      'appearance': '짧은 머리와 네모진 턱을 가진 근육질의 모습을 가진 사람'},
     {'height': 175, 'weight': 75,
      'appearance': '날씬하고 날렵한 이목구비와 안경 쓴 모습'},
     {'height': 168, 'weight': 60,
      'appearance': '동그란 얼굴에 안경을 쓴 젊은 모습'},
     {'height': 180, 'weight': 80,
      'appearance': '키가 크고 날렵한 턱선이 잘 생긴 얼굴'},
     {'height': 173, 'weight': 68,
      'appearance': '단정한 머리와 잘 다듬어진 수염으로 스타일리시한 사람'},
     {'height': 178, 'weight': 73,
      'appearance': '날씬한 몸매와 튀어나온 광대뼈가 잘 어울리는 사람'},
     {'height': 171, 'weight': 67,
      'appearance': '깨끗하고 따뜻한 미소와 차분한 태도를 가진 사람'},
     {'height': 176, 'weight': 72,
      'appearance': '날렵한 드레스 센스와 깔끔한 면도로 자신감이 있는 사람'
      },
     {'height': 179, 'weight': 78,
      'appearance': '강한 근육질 체격과 날카로운 이목구비를 갖춘 사람'},
     {'height': 167, 'weight': 63,
      'appearance': '친절한 미소와 친근한 분위기의 발랄한 사람'
      },
     {'height': 172, 'weight': 69,
      'appearance': '그루터기로 튼튼하고 캐주얼하고 여유로운 스타일을 가진 사람'
      },
     ]

asian_women_looks = [
     {'height': 160, 'weight': 50,
      'appearance': '가늘고 섬세한 이목구비와 긴 흑발을 가진 사람'
      },
     {'height': 165, 'weight': 55,
      'appearance': '키가 크고 날씬하며 하트 모양의 얼굴과 큰 눈을 가진 사람'
      },
     {'height': 155, 'weight': 45,
      'appearance': '뽀얀 피부, 장밋빛 볼, 작은 코를 가진 사람'
      },
     {'height': 162, 'weight': 50,
      'appearance': '잘록한 허리와 높은 광대뼈, V자 턱선으로 우아한 모습'
      },
     {'height': 158, 'weight': 52,
      'appearance': '가느다란 목과 가느다란 손가락, 부드러운 미소로 우아한 모습'
      },
     {'height': 163, 'weight': 53,
      'appearance': '작은 코, 긴 속눈썹, 부드러운 목소리로 여성스러운 모습'
      },
     {'height': 166, 'weight': 57,
      'appearance': '긴 다리, 도톰한 입술, 당당한 걸음걸이를 가진 조각상 같은 모습'
      },
     {'height': 159, 'weight': 49,
      'appearance': '동그란 볼살에 발랄한 성격에 귀여운 외모를 가진 사람'
      },
     {'height': 161, 'weight': 51,
      'appearance': '잘록한 허리, 가느다란 팔, 조용한 태도를 가진 사람'
      },
     {'height': 167, 'weight': 60,
      'appearance': '장엄한 태도, 강한 이목구비, 압도적인 존재감을 지닌 장엄한 모습'
      },
     {'height': 154, 'weight': 48,
      'appearance': '달콤한 미소와 작은 귀, 다정한 성품이 사랑스러운 사람'
      },
     {'height': 164, 'weight': 54,
      'appearance': '세련된 외모, 예리한 위트, 예리한 스타일 감각으로 세련된 사람'
      },
     ]

wealth = [
     {"properties": "부동산 7,500만원, 보증금 2,000만원, 주식 1,000만원, 명품 500만원"},
     {"properties": "부동산 1억 5천만 원, 보증금 3천만 원, 주식 2천만 원, 미술품 천만 원"},
     {"properties": "부동산 2억 5천만 원, 보증금 5천만 원, 주식 3천만 원, 보석류 2천만 원"},
     {"properties": "부동산 3억원, 보증금 7,500만원, 주식 4,000만원, 골동품 자동차 2,500만원"},
     {"properties": "부동산 4억원, 보증금 1억원, 주식 5천만원, 고급 와인 3천만원"},
     {"properties": "부동산 5억원, 보증금 1억 2,500만원, 주식 6,000만원, 수집용 피규어 3,500만원"},
     {"properties": "부동산 6억 원, 보증금 1억 5천만 원, 주식 7천만 원, 명품 의류 4천만 원"},
     {"properties": "부동산 7억원, 보증금 1억 7,500만원, 주식 8,000만원, 고급시계 4,500만원"},
     {"properties": "부동산 8억원, 예금 2억원, 주식 9천만원, 희귀도서 5천만원"},
     {"properties": "부동산 9억원, 보증금 2억2500만원, 주식 1억원, 자동차 5500만원"},
     {"properties": "부동산 9억 5천만 원, 보증금 2억 5천만 원, 주식 1억 5천만 원, 요트 6천만 원"},
     {"properties": "부동산 10억원, 예금 3억원, 주식 2억원, 미술품 1억원"},
]


sep_index = tokenizer.additional_special_tokens.index('<|sep|>')
sep_token_id = tokenizer.additional_special_tokens_ids[sep_index]
tt = tokenizer("\n?.")
newline_token_id = tt['input_ids'][0]
question_mark_token_id = tt['input_ids'][1]
period_token_id = tt['input_ids'][2]
print(f'sep_token_id={sep_token_id}\nnewline_token_id={newline_token_id}\nquestion_mark_token_id={question_mark_token_id}\nperiod_token_id={period_token_id}')
print(tokenizer.decode([224]))

def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    def command_func(update, context, *args, **kwargs):
        context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
        return func(update, context,  *args, **kwargs)

    return command_func

def start(update: Update, context: CallbackContext):
	update.message.reply_text(HELP_TEXT)

def help(update: Update, context: CallbackContext):
	update.message.reply_text(HELP_TEXT)

def clear_chat_history(context):
    context.user_data['chat_history'] = {"normalmode":[], "testmode":[]}
    return

def query(context, user_input):
    if context.user_data['councelor_type'] == "chatting":
        return chat_query(context, user_input, chat_prompt_normal)
    elif context.user_data['councelor_type'] == "therapist":
        return chat_query(context, user_input, chat_prompt_therapist)
    elif context.user_data['councelor_type'] == "doctor":
        return chat_query(context, user_input, chat_prompt_doctor)
    elif context.user_data['councelor_type'] == "expert":
        if not user_input.endswith(('?', ".", "!")):
            user_input = user_input + "?"
        return chat_query(context, user_input, chat_prompt_expert, "B", "A", 3, 120)
    elif context.user_data['councelor_type'] == "expert2":
        if not user_input.endswith(('?', ".", "!")):
            user_input = user_input + "?"
        return chat_query(context, user_input, chat_prompt_expert2, "B", "A", 3, 120)
    elif context.user_data['councelor_type'] == "mbti":
        return chat_query(context, user_input, chat_prompt_mbti, "B", "A", 6)
    elif context.user_data['councelor_type'] == "fortune":
        return chat_query(context, user_input, context.user_data["prompt"], "B", "A", 2, 120)
    elif context.user_data['councelor_type'] == "prompt":
        return prompt_query(context, user_input)
        
def skip_eos_token(output):
    for index, item in enumerate(output):
        if item != tokenizer.eos_token_id:
            output = output[index:]
            print(f'skip eos token={index}')
            break
    return output

def generate(context, contents, chat_mode = False, open_end = False, gen_len = 0):
    global generator
    contents = contents.strip()
    if not open_end:
        contents = f'{contents}<|sep|>'
    encoded_input = tokenizer(contents, return_tensors='pt').to(device)
    print(f"text={len(contents)}, token={encoded_input['input_ids'].size()}")
    input_length = encoded_input['input_ids'].size()[1]
    print(f'input_length={input_length}')
    if gen_len > 0:
        max_length = input_length + gen_len
    else:
        if not chat_mode:
            if input_length * 2 + 10 < max_output_length:
                max_length = input_length * 2 + 10
                if max_length < min_output_length:
                    max_length = min_output_length
            else:
                max_length = max_output_length
        else:
            max_length = input_length + 50
    print(f'max_length={max_length}')
    
    try:
        testmode = False
        if 'mode' not in context.user_data or context.user_data['mode'] == "normalmode":
            model = gpt
            print(f'running on normal model, checkpoint={checkpoint}.')
        else:
            testmode = True
            model = gpt_on_test
            print(f'running on test model, checkpoint={checkpoint_test}.')
        
        start_time = datetime.today().timestamp()
        if generator is not None and deepspeed_mode and testmode:
            result_id = generator.query_non_block(
                {"query": [contents]}, 
                do_sample=False, 
                max_new_tokens=120,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                num_beams=3,       # DeepSpeed not support num_beam > 1
                length_penalty=1.0,
                temperature=0.6,
                top_k=4,
                top_p=0.6,
                no_repeat_ngram_size=3, 
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                #begin_suppress_tokens=[tokenizer.eos_token_id, sep_token_id, newline_token_id, question_mark_token_id, period_token_id],
            )
            result = None
            for i in range(100):
                result = generator.get_pending_task_result(result_id)
                if result is not None:
                    result = result.response[0]
                    break
                time.sleep(1)
            if result is None:
                generated = "음... 뭔가 잘 못 됐어..."
            else:
                generated = result
            prompt = contents
            print(generated)
            generated = generated[len(contents):]
            garbage = ""
        else:
            if False:
                output_sequences = model.generate(
                    encoded_input["input_ids"], 
                    do_sample=False,
                    early_stopping=True,
                    num_beams=3,
                    length_penalty=1.0,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=3, 
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=[tokenizer.eos_token_id, sep_token_id],
                    begin_suppress_tokens=[tokenizer.eos_token_id, sep_token_id, newline_token_id, question_mark_token_id, period_token_id],
                    max_length=max_length
                )
            else:
                output_sequences = model.generate(
                    encoded_input["input_ids"], 
                    do_sample=False,
                    early_stopping=True,
                    num_beams=3,
                    length_penalty=1.0,
                    temperature=0.6,
                    top_k=4,
                    top_p=0.6,
                    no_repeat_ngram_size=3, 
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=[tokenizer.eos_token_id, sep_token_id],
                    begin_suppress_tokens=[tokenizer.eos_token_id, sep_token_id, newline_token_id, question_mark_token_id, period_token_id],
                    max_length=max_length
                )
            output = output_sequences[0].tolist()
            
            prompt = tokenizer.decode(output[:input_length], skip_special_tokens=False)
            output = output[input_length:]
            print(output)
            output = skip_eos_token(output)
            try:
                stop = output.index(tokenizer.eos_token_id)
            except:
                stop = len(output)
            generated = tokenizer.decode(output[:stop], skip_special_tokens=False).strip()
            garbage = tokenizer.decode(output[stop:], skip_special_tokens=False)        
        end_time = datetime.today().timestamp()
        print(f"******inference time = {end_time-start_time}")
        print(f'prompt={prompt}\ngenerated={generated}')
        generated = generated.replace("답은 아래와 같습니다.\n", "")        
        generated = generated.replace("답변:", "").strip()
        generated = generated.replace("키키", "ㅋㅋ")
        print(f'\n\ngarbage={garbage}')        
    except Exception as e:
        print(f'generate error = {e}')
        traceback.print_exc()
        prompt = "error!"
        generated = "음..."
    print(f'final generation={generated}')
    
    return prompt, generated
    
def prompt_query(context, user_input):
    content = f"{user_input}"
    prompt, generated = generate(context, content, False, True)
    return prompt, generated
        
def chat_query(context, user_input, chat_prompt, user_prefix="B", bot_prefix="A", MAX_CHAT_HISTORY=7, CHAT_RESPONSE_LEN=120):
    chat_history = context.user_data['chat_history'][context.user_data['mode']]
    contents = chat_prompt
    last_bot_message = None
    now = datetime.today().timestamp()
    duplicated = next((item for item in chat_history if item["user"] == user_input), None)
    if duplicated is not None:
        chat_history.remove(duplicated)
    for ch in chat_history:
        ch_time = int(now - ch['time'])
        contents += f"{user_prefix}: {ch['user']}\n"
        last_bot_message = ch['bot']
        if last_bot_message is not None:
            contents += f'{bot_prefix}: {last_bot_message}\n'
    user_input = user_input.strip()
    contents += f"{user_prefix}: {user_input}\n{bot_prefix}: "

    prompt, generated = generate(context, contents, True, True, CHAT_RESPONSE_LEN)

    match = re.search('\n?[A-Z]\s?(?:[:;-]|$)', generated)
    if match is None:
        bot_message = generated
    else:
        stop_index = match.start()
        bot_message = generated[:stop_index].strip()
        print(f'prefix stop remain = {generated[stop_index:]}')
    bot_message_in_history = bot_message
    if bot_message == last_bot_message:
        bot_message_in_history = None
    timestamp = datetime.today().timestamp()
            
    chat_history.append({"user": user_input, "bot": bot_message_in_history, "time": timestamp})
    while len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
    print(f"bot_message={bot_message}")
    return prompt, bot_message

def try_parse_datetime(date_str, format_str):
    try:
        return datetime.strptime(date_str, format_str)
    except:
        return None    

def parse_date_input(date_str):
    if date_str == '11':
        date_str = '1966/2/26 오후 3:00'
    if date_str == '22':
        date_str = '1990/11/22 오후 7:00'
    date_str = date_str.replace('오전', ' AM ')
    date_str = date_str.replace('오후', ' PM ')
    date_str = re.sub(r'[^\dAPM:]+',' ', date_str)
    date_str = ' '.join(date_str.split())
    date_str = date_str.strip().upper()
    try:
        datetime_object = parse(date_str)
    except:
        print(f'date paring failed={date_str}')
        datetime_object = try_parse_datetime(date_str, "%Y %m %d %p %I %M")
        if datetime_object is None:
            datetime_object = try_parse_datetime(date_str, "%Y %m %d %p %I")
        if datetime_object is None:
            datetime_object = try_parse_datetime(date_str, "%Y %m %d %H")
        if datetime_object is None:
            datetime_object = try_parse_datetime(date_str, "%Y %m %d %H %M")
        if datetime_object is None:
            print("invalid date input, input failed")
    return datetime_object

def build_fortune_text(birtyday: datetime, sex):
    print(birtyday, birtyday.day)
    job = job_list[birtyday.day % 7]
    job = ",".join(job)
    personality_index = int(birtyday.day * birtyday.hour) % 20 
    Personality_type = Personality_types[personality_index]
    personality = f"{Personality_type['type']} 성격으로, {Personality_type['description']}"
    meet_when_month = int(birtyday.day * birtyday.year) % 12 + 1
    meet_when = f"다가오는 {meet_when_month}월 일 가능성이 높아."
    meet_where_index = int(birtyday.day * birtyday.month) % 10
    meet_where = places_to_meet[meet_where_index]
    meet_where = ",".join(meet_where)
    meet_where = f"{meet_where} 중 하나가 될 가능성이 있다."
    appearance_index = int(birtyday.month * birtyday.hour) % 12
    if sex == 'male':
        looks = asian_women_looks 
        sex_str = "남자" 
        sex_partner_str = "여자"
        sex_partner_str2 = "그녀"
    else:
        looks = asian_man_looks
        sex_str = "여자" 
        sex_partner_str = "남자"
        sex_partner_str2 = "그분"
    appearance = f"키는 {looks[appearance_index]['height']}센티미터 이고, 몸무게는 {looks[appearance_index]['weight']} 이며, {looks[appearance_index]['appearance']} 정도로 보여"
    money_index = int(birtyday.day * birtyday.hour) % 12
    money = f"{wealth[money_index]['properties']} 정도로 예상된다."
    fortune_prompt = f"""
A는 점을 봐주는 점쟁이이다. 
B는 점을 보러온 고객이고 {sex_str}인데 앞으로 만날 {sex_partner_str}에 대해서 궁금해서 점을 보러 왔다. 
B의 모든 질문은 미래 애인에 대한 것이다. 절대 본인 즉 점쟁이 A의 이야기를 하지 마시오.
B의 미래의 애인 또는 장차 만나게 될 사람, 미지의 그 사람의 정보는 다음과 같다.
{sex_partner_str2}의 성별은 {sex_partner_str} 이다.
{sex_partner_str2}의 직업은 {job} 중 하나일 가능성이 높다.
{sex_partner_str2}의 성격은 {personality} 일 가능성이 있어.
{sex_partner_str2}을 만나는 시기는 {meet_when}.
{sex_partner_str2}을 만나는 장소는 {meet_where}.
{sex_partner_str2}의 외모는 {appearance}.
{sex_partner_str2}의 재산은 {money}.
위 내용에 없는 것을 답할 경우에는 불확실한 추정임을 반드시 이야기 해야 하며 절대 점쟁이 본인 즉 A의 이야기는 하지 마시오.
답변시에는 명리학에 기반한 추정일 뿐임을 꼭 이야기 하시오.
위 내용에 기반하여 점쟁이로서 성실한 자세, 약간 신들린 모습으로, 고객의 질문에 답하시오. 
B: 그 사람의 성격은 어때?
A: 당신이 만날 미래 애인은 {personality} 일 가능성이 높아보여.
"""
    print(fortune_prompt)
    return fortune_prompt

def chatting(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "chatting"  
    init_user_data(context)  
    update.message.reply_text("일반 채팅 모드로 전환 되었습니다.")

def therapist(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "therapist"  
    init_user_data(context)  
    update.message.reply_text("심리 상담사 채팅 모드로 전환 되었습니다.")

def doctor(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "doctor"  
    init_user_data(context)  
    update.message.reply_text("응급 의사 채팅 모드로 전환 되었습니다.")

def prompt(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "prompt"  
    init_user_data(context)  
    update.message.reply_text("prompt 모드로 전환 되었습니다.")

def expert(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "expert"  
    init_user_data(context)  
    update.message.reply_text("expert 채팅 모드로 전환 되었습니다..")

def expert2(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "expert2"  
    init_user_data(context)  
    update.message.reply_text("expert2 친근 채팅 모드로 전환 되었습니다..")

def mbti(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "mbti"  
    init_user_data(context)  
    update.message.reply_text("mbti 모드로 전환 되었습니다.")

def fortune(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "fortune"
    init_user_data(context)  
    context.user_data.pop("birthday", None)
    context.user_data.pop("birthday_confirm", None)
    context.user_data.pop("sex", None)
    update.message.reply_text("역술가 모드로 전환 되었습니다.")
    update.message.reply_text("생년월일과 출생시간을 입력 해. 1980년 3월 20일 오후 2시 20분 또는 1999.2.12 22:00, 1988/12/31 오후 1:30, 198003200220 같은 형식으로 하면 돼.")

def testmode(update: Update, context: CallbackContext):
    context.user_data["mode"] = "testmode"  
    clear_chat_history(context)
    update.message.reply_text("test 모드로 전환 되었습니다")

def normalmode(update: Update, context: CallbackContext):
    context.user_data["mode"] = "normalmode"  
    clear_chat_history(context)
    update.message.reply_text("normal 모드로 전환 되었습니다")
    
def shownormal(update: Update, context: CallbackContext):
    if "shownormal" not in context.user_data:
        context.user_data["shownormal"] = False
    context.user_data["shownormal"] = not context.user_data["shownormal"]  
    update.message.reply_text(f"show normal model = {context.user_data['shownormal']}")

def status(update: Update, context: CallbackContext):
    if 'mode' not in context.user_data:
        context.user_data['mode'] = 'normalmode'
    if 'councelor_type' not in context.user_data:
        context.user_data['councelor_type'] = 'expert'
    s = f"runmode = {context.user_data['mode']}\nresponse type={context.user_data['councelor_type']}\nshow normal={context.user_data['shownormal']}"  
    clear_chat_history(context)
    update.message.reply_text(s)

def clear_chat_history_handler(update: Update, context: CallbackContext):
    clear_chat_history(context)
    update.message.reply_text("채팅 히스토리가 삭제 되었습니다.")

def send_typing(context, chat_id):
    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
  
def build_menu(buttons, n_cols, header_buttons=None, footer_buttons=None):
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, header_buttons)
    if footer_buttons:
        menu.append(footer_buttons)
    return menu
  
def keyboard_callback(update: Update, context: CallbackContext) :
    data_selected = update.callback_query.data
    print("callback : ", data_selected)
    if data_selected in ["male", "female"]:
        context.user_data['sex'] = data_selected
        birtyday = context.user_data["birthday"]
        sex = context.user_data["sex"]
        prompt = build_fortune_text(birtyday, sex)
        context.user_data["prompt"] = prompt
        partner = "여친" if data_selected == 'male' else "남친"
        update.callback_query.message.reply_text(f"좋아 준비가 다 됐어. 미래 {partner}에 대해서 뭐든 물어봐.")
    elif data_selected in ["birthday_yes", "birthday_no"]:
        if data_selected == "birthday_yes":
            context.user_data["birthday_confirm"] = True
            if "sex" not in context.user_data:
                show_list = []
                show_list.append(InlineKeyboardButton("남자", callback_data="male")) 
                show_list.append(InlineKeyboardButton("여자", callback_data="female")) 
                show_markup = InlineKeyboardMarkup(build_menu(show_list, len(show_list))) 
                update.callback_query.message.reply_text("본인 성별이 뭐야?", reply_markup=show_markup)
                return
        else:
            context.user_data.pop("birthday", None)
            update.callback_query.message.reply_text("생년월일과 출생시간을 입력 해. 1980년 3월 20일 오후 2시 20분 또는 1999.2.12 22:00, 1988/12/31 오후 1:30, 198003200220 같은 형식으로 하면 돼.")

def init_user_data(context: CallbackContext):
    if "councelor_type" not in context.user_data:
        context.user_data["councelor_type"] = "expert"
    if "mode" not in context.user_data:
        context.user_data["mode"] = "normalmode"
    if "shownormal" not in context.user_data:
        context.user_data["shownormal"] = False  
    clear_chat_history(context)
                
def unknown(update: Update, context: CallbackContext):
    #print(update)
    now = datetime.today()
    if update.message is not None:
        message = update.message
    elif update.edited_message is not None:
        message = update.edited_message
    username = message.chat['username']
    first_name = message.chat['first_name']
    q = message.text
    q = q.strip()

    print(f"\n\n---------------\n{now} {first_name}({username}): {q}\n")
    if "councelor_type" not in context.user_data or "mode" not in context.user_data:
        context.user_data["councelor_type"] = "expert"
        init_user_data(context)
        if username == 'ninedra9ons':
            context.user_data["mode"] = "testmode"
            #context.user_data["shownormal"] = True

        message.reply_text(f"현재 {context.user_data['councelor_type']} 모드입니다. 가능한 명령을 보려면 /help 를 치세요.")
        message.reply_text("저사양 GPU에서 동작중이라 응답속도가 느립니다. 긴 문장 생성에는 10초 이상이 걸릴 수도 있습니다.")
        message.reply_text("Language model restarted.")
        # update.message.reply_text(HELP_TEXT)
        if username == 'ninedra9ons':
            status(update, context)

    councelor_type = context.user_data["councelor_type"]
    if councelor_type == "fortune":
        if "birthday" not in context.user_data:
            birtyday = parse_date_input(q)
            if birtyday is not None:
                print(f'birthday input successful={birtyday}')
                context.user_data["birthday"] = birtyday
                show_list = []
                show_list.append(InlineKeyboardButton("맞아", callback_data="birthday_yes")) 
                show_list.append(InlineKeyboardButton("틀려", callback_data="birthday_no")) 
                show_markup = InlineKeyboardMarkup(build_menu(show_list, len(show_list))) 
                birthday_str = context.user_data["birthday"].strftime('%Y년 %m월 %d일 %I시 %M분 %p')
                message.reply_text(f"생시가 {birthday_str}, 이거 맞나 확인해?", reply_markup=show_markup)
                return
            else:
                message.reply_text("생일을 입력 해야 해. 안그러면 진행이 안돼.")
                message.reply_text("생년월일과 출생시간을 입력 해. 1980년 3월 20일 오후 2시 20분 또는 1999.2.12 22:00, 1988/12/31 오후 1:30, 198003200220 같은 형식으로 하면 돼.")
                return
                    
    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    t = Timer(8, send_typing, [context, update.effective_message.chat_id])  
    t.start()  
    
    prompt, a = query(context, q)
    a = a.strip()
    print(f'query result="{a}", len={len(a)}')
    if len(a) == 0 or prompt=='error!':
        a = "음..."
        clear_chat_history(context)
        print('no generation, retry with clear chat history.')
        message.reply_text("잠깐만...")
        prompt, a = query(context, q)
        a = a.strip()
    t.cancel()

    print(f'query result="{a}", len={len(a)}')
    if len(a) > 0 and prompt!='error!':
        message.reply_text(a)
    
    if "mode" not in context.user_data.keys():
        context.user_data['mode'] = "normalmode" 
    if "shownormal" not in context.user_data.keys():
        context.user_data['shownormal'] = False 
    if context.user_data['mode'] == "testmode" and context.user_data["shownormal"]:
        context.user_data['mode'] = "normalmode"
        prompt, a2 = query(context, q)
        a2 = a2.strip()
        context.user_data['mode'] = "testmode"
        message.reply_text(f'{a2}-[N]')
        
def unknown_text(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry I can't recognize you , you said '%s'" % update.message.text)

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(CommandHandler('clear', clear_chat_history_handler))

updater.dispatcher.add_handler(CommandHandler('prompt', prompt))
updater.dispatcher.add_handler(CommandHandler('chatting', chatting))
updater.dispatcher.add_handler(CommandHandler('therapist', therapist))
updater.dispatcher.add_handler(CommandHandler('doctor', doctor))
updater.dispatcher.add_handler(CommandHandler('expert', expert))
updater.dispatcher.add_handler(CommandHandler('expert2', expert2))
updater.dispatcher.add_handler(CommandHandler('mbti', mbti))
updater.dispatcher.add_handler(CommandHandler('fortune', fortune))

updater.dispatcher.add_handler(CommandHandler('testmode', testmode))
updater.dispatcher.add_handler(CommandHandler('normalmode', normalmode))

updater.dispatcher.add_handler(CommandHandler('shownormal', shownormal))

updater.dispatcher.add_handler(CommandHandler('status', status))

updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown, run_async=True))
updater.dispatcher.add_handler(MessageHandler(
	Filters.command, unknown)) # Filters out unknown commands

# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))
updater.dispatcher.add_handler(CallbackQueryHandler(keyboard_callback))

updater.start_polling()

print("Ready!")

