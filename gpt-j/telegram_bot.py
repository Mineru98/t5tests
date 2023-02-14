from functools import wraps
import os, re
from threading import Timer   

from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from telegram import (ChatAction)
from telegram.ext.dispatcher import run_async

from transformers import AutoTokenizer, logging, pipeline, AutoModelForCausalLM
import torch

checkpoint = 1220
#latest_model_dir = "EleutherAI/polyglot-ko-1.3b"
latest_model_dir = f"/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func/checkpoint-{checkpoint}"
latest_model_dir_on_test = "/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func/checkpoint-1360"

HELP_TEXT = f"""
Large Language Model chat-bot by Sempahore. V 0.1 checkpoint-{checkpoint}
3.8B parameters language model, 1/46 of chatGPT in parameter size.
Internal experimental release.
현재 고물 컴퓨터에서 실행 중이므로 긴 문장 생성시 응답 속도가 10초 이상 걸립니다. 

명령어.
/chatting - 일반 잡담 채팅, 사람을 가정하고 하는 채팅. 주제는 제한 없음.
/expert - 백과 사전식 질의 응답.
/doctor
/therapist

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
A의 이름, 직업, 나이등 기타 신상 정보는 모두 특급 비밀이다.
위 내용에 기반하여 이전 대화 내용을 우선으로 성실한 해당 분야 전문가로서, 질문에 답하시오.
B: 하늘이 푸른 이유는?
A: 빛이 대기를 통과하면서 파장이 짧은 푸른빛은 산란되고, 파장이 긴 붉은빛은 대기에 흡수되기 때문이지.
B: 선거의 4원칙은?
A: 보통선거, 평등선거, 직접선거, 비밀선거가 있어.
"""

max_output_length = 1024
min_output_length = 512
CHAT_RESPONSE_LEN = 120

tokenizer_dir = latest_model_dir
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

updater = Updater(os.environ['TELEGRAM_LM_CHAT'], use_context=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
gpt = AutoModelForCausalLM.from_pretrained(
    latest_model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device, torch.float16)

gpt_on_test = AutoModelForCausalLM.from_pretrained(
    latest_model_dir_on_test,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device, torch.float16)

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
    print(f"\n\n\nstart new query----\n{user_input}\n")
    if context.user_data['councelor_type'] == "chatting":
        return chat_query(context, user_input, chat_prompt_normal)
    elif context.user_data['councelor_type'] == "therapist":
        return chat_query(context, user_input, chat_prompt_therapist)
    elif context.user_data['councelor_type'] == "doctor":
        return chat_query(context, user_input, chat_prompt_doctor)
    elif context.user_data['councelor_type'] == "expert":
        return chat_query(context, user_input, chat_prompt_expert, "B", "A", 3)
    elif context.user_data['councelor_type'] == "mbti":
        return chat_query(context, user_input, chat_prompt_mbti, "B", "A", 6)
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
        if 'mode' not in context.user_data or context.user_data['mode'] == "normalmode":
            model = gpt
            print('running on normal model.')
        else:
            model = gpt_on_test
            print('running on test model.')
        output_sequences = model.generate(
            encoded_input["input_ids"], 
            do_sample=False,
            early_stopping=True,
            num_beams=5,
            length_penalty=1.0,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3, 
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, sep_token_id],
            begin_suppress_tokens=[tokenizer.eos_token_id, sep_token_id, newline_token_id, question_mark_token_id, period_token_id],
            # forced_eos_token_id=tokenizer.eos_token_id,
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
        print(f'prompt={prompt}\ngenerated={generated}')
        generated = generated.replace("답은 아래와 같습니다.\n", "")        
        generated = generated.replace("답변:", "").strip()
        generated = generated.replace("키키", "ㅋㅋ")
        print(f'\n\ngarbage={garbage}')        
    except Exception as e:
        print(f'generate error = {e}')
        prompt = "error!"
        generated = "음..."
    print(f'final generation={generated}')
    
    return prompt, generated
    
def prompt_query(context, user_input):
    content = f"{user_input}"
    prompt, generated = generate(context, content, False, True)
    return prompt, generated
        
def chat_query(context, user_input, chat_prompt, user_prefix="B", bot_prefix="A", MAX_CHAT_HISTORY=7):
    chat_history = context.user_data['chat_history'][context.user_data['mode']]
    contents = chat_prompt
    last_bot_message = None
    for ch in chat_history:
        contents += f"{user_prefix}: {ch['user']}\n"
        last_bot_message = ch['bot']
        if last_bot_message is not None:
            contents += f"{bot_prefix}: {last_bot_message}\n"
    user_input = user_input.strip()
    contents += f"{user_prefix}: {user_input}\n{bot_prefix}: "

    prompt, generated = generate(context, contents, True, True, CHAT_RESPONSE_LEN)

    match = re.search('\n[A-Z][:;]', generated)
    if match is None:
        bot_message = generated
    else:
        stop_index = match.start()
        bot_message = generated[:stop_index].strip()
    bot_message_in_history = bot_message
    if bot_message == last_bot_message:
        bot_message_in_history = None
    chat_history.append({"user": user_input, "bot": bot_message_in_history})
    while len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
    print(f"bot_message={bot_message}")
    return prompt, bot_message

def chatting(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "chatting"  
    clear_chat_history(context)
    update.message.reply_text("일반 채팅 모드로 전환 되었습니다.")

def therapist(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "therapist"  
    clear_chat_history(context)
    update.message.reply_text("심리 상담사 채팅 모드로 전환 되었습니다.")

def doctor(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "doctor"  
    clear_chat_history(context)
    update.message.reply_text("응급 의사 채팅 모드로 전환 되었습니다.")

def prompt(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "prompt"  
    clear_chat_history(context)
    update.message.reply_text("prompt 모드로 전환 되었습니다.")

def expert(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "expert"  
    clear_chat_history(context)
    update.message.reply_text("expert 채팅 모드로 전환 되었습니다..")

def mbti(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "mbti"  
    clear_chat_history(context)
    update.message.reply_text("mbti 모드로 전환 되었습니다.")

def testmode(update: Update, context: CallbackContext):
    context.user_data["mode"] = "testmode"  
    clear_chat_history(context)
    update.message.reply_text("test 모드로 전환 되었습니다")

def normalmode(update: Update, context: CallbackContext):
    context.user_data["mode"] = "normalmode"  
    clear_chat_history(context)
    update.message.reply_text("normal 모드로 전환 되었습니다")
    
def shownormal(update: Update, context: CallbackContext):
    context.user_data["shownormal"] = not context.user_data["shownormal"]  
    update.message.reply_text(f"show normal model = {context.user_data['shownormal']}")

def status(update: Update, context: CallbackContext):
    if 'mode' not in context.user_data:
        context.user_data['mode'] = 'normalmode'
    if 'councelor_type' not in context.user_data:
        context.user_data['councelor_type'] = 'chatting'
    s = f"runmode = {context.user_data['mode']}\nresponse type={context.user_data['councelor_type']}"  
    clear_chat_history(context)
    update.message.reply_text(s)

def clear_chat_history_handler(update: Update, context: CallbackContext):
    clear_chat_history(context)
    update.message.reply_text("채팅 히스토리가 삭제 되었습니다.")

def send_typing(context, chat_id):
    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
  
def unknown(update: Update, context: CallbackContext):
    if "councelor_type" not in context.user_data.keys():
        context.user_data["councelor_type"] = "expert"
        context.user_data["mode"] = "normalmode"
        context.user_data["shownormal"] = False  
        clear_chat_history(context)

        update.message.reply_text("현재 전문가 질의 응답 모드입니다. 가능한 명령을 보려면 /help 를 치세요.")
        update.message.reply_text("저사양 GPU에서 동작중이라 응답속도가 느립니다. 긴 문장 생성에는 10초 이상이 걸릴 수도 있습니다.")
        update.message.reply_text("Language model restarted.")
        # update.message.reply_text(HELP_TEXT)
        
    q = update.message.text
    q = q.strip()

    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    t = Timer(8, send_typing, [context, update.effective_message.chat_id])  
    t.start()  
    
    prompt, a = query(context, q)
    a = a.strip()
    if a is None or len(a) == 0 or prompt=='error!':
        a = "음..."
        clear_chat_history(context)
        print('no generation, retry with clear chat history.')
        prompt, a = query(context, q)
        a = a.strip()
    t.cancel()

    print(f'query result="{a}", len={len(a)}')
    update.message.reply_text(a)
    
    if context.user_data['mode'] == "testmode" and context.user_data["shownormal"]:
        context.user_data['mode'] = "normalmode"
        prompt, a2 = query(context, q)
        a2 = a2.strip()
        context.user_data['mode'] = "testmode"
        update.message.reply_text(f'{a2}-[N]')
        
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
updater.dispatcher.add_handler(CommandHandler('mbti', mbti))

updater.dispatcher.add_handler(CommandHandler('testmode', testmode))
updater.dispatcher.add_handler(CommandHandler('normalmode', normalmode))

updater.dispatcher.add_handler(CommandHandler('shownormal', shownormal))

updater.dispatcher.add_handler(CommandHandler('status', status))

updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown, run_async=True))
updater.dispatcher.add_handler(MessageHandler(
	Filters.command, unknown)) # Filters out unknown commands

# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

updater.start_polling()

print("Ready!")