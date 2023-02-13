from functools import wraps
import os
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

HELP_TEXT = """
언어모델 챗 봇 by Sempahore. 3.8B parameters language model, 1/46 of chatGPT.
현재 고물 컴퓨터에서 실행 중이므로 긴 문장 생성시 응답 속도가 10초 이상 걸립니다. 

명령어.
/chatting - 일반 잡담 채팅, 35세 직장인 남성을 가정하고 하는 채팅임. 사람을 가정하고 하는 채팅. 주제는 제한 없음.
/clear - 채팅 히스토리 삭제
/qna - 질의 응답, 질문에 대해 답을 하며, 이전 질문/답과 연결되지 않음.
/mqna - 다중 질의 응답, 채팅식으로 문답을 이어 나갈 수 있음.
/doctor
/therapist
/prompt - 기타 프롬프트 입력, 일반 문장 입력시 해당 문장을 시작으로 문장을 연속해서 만들어 냄.
기능으로 동작하는 프롬프트도 있는데 채팅, qna등이 모두 기능 프롬프트로 구현된 것임. 아래는 프롬프터 예제.

- 번역
<한글문장> 
영어로 번역 하시오. 
<영어문장> 
한글로 번역하시오. 

- 기사작성
다음 문장을 주제로 기사를 작성 하시오. 
<기사제목> 

- 요약
<장문의 글>
위글을 요약 하시오.

- Q&A
<장문의 글>
위글을 보고 아래 질문에 답하시오.
<질문>
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
"""

max_output_length = 1024
min_output_length = 512

#latest_model_dir = "EleutherAI/polyglot-ko-1.3b"
latest_model_dir = "/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func/checkpoint-1060"
latest_model_dir_on_test = "/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func-unfreeze16/checkpoint-000"

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
    context.user_data["chat_history"] = []
    return

def query(context, user_input):
    print(f"\n\n\nstart new query----\n{user_input}\n")
    if context.user_data['councelor_type'] == "chatting":
        return chat_query(context, user_input, chat_prompt_normal)
    elif context.user_data['councelor_type'] == "therapist":
        return chat_query(context, user_input, chat_prompt_therapist)
    elif context.user_data['councelor_type'] == "doctor":
        return chat_query(context, user_input, chat_prompt_doctor)
    elif context.user_data['councelor_type'] == "qna":
        return qna_query(context, user_input)
    elif context.user_data['councelor_type'] == "mqna":
        return mqna_query(context, user_input)
    elif context.user_data['councelor_type'] == "prompt":
        return prompt_query(context, user_input)
        
def skip_eos_token(output):
    for index, item in enumerate(output):
        if item != tokenizer.eos_token_id:
            output = output[index:]
            print(f'skip eos token={index}')
            break
    return output

def generate(context, contents, chat_mode = False, open_end = False):
    contents = contents.strip()
    if not open_end:
        contents = f'{contents}<|sep|>'
    encoded_input = tokenizer(contents, return_tensors='pt').to(device)
    print(f"text={len(contents)}, token={encoded_input['input_ids'].size()}")
    input_length = encoded_input['input_ids'].size()[1]
    print(f'input_length={input_length}')
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
        prompt = ""
        generated = "음..."
    print(f'final generation={generated}')
    
    return prompt, generated
    
def prompt_query(context, user_input):
    content = f"{user_input}"
    prompt, generated = generate(context, content, False, True)
    return generated
        
def qna_query(context, user_input):
    content = f"전문가로서 아래 질문에 답하시오.\n{user_input}"
    prompt, generated = generate(context, content)
    return generated

def mqna_query(context, user_input):
    MAX_CHAT_HISTORY = 3
    user_prefix = "Q"
    bot_prefix = "A"

    chat_history = context.user_data['chat_history']
    contents = ""
    for ch in chat_history:
        contents += f"{user_prefix}: {ch['user']}\n{bot_prefix}: {ch['bot']}\n"
    user_input = user_input.strip()
    contents += f"질문에 답하시오. 질문이 위 내용과 관련 없으면 위 내용은 완전히 무시하고 답하시오.\n{user_input}"
    
    prompt, generated = generate(context, contents)

    bot_message = generated.replace("\n", " ")
    chat_history.append({"user": user_input, "bot": bot_message})
    while len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
    print(f"bot_message={bot_message}")
    return bot_message
        
def chat_query(context, user_input, chat_prompt, user_prefix="B", bot_prefix="A"):
    MAX_CHAT_HISTORY = 7

    chat_history = context.user_data['chat_history']
    contents = chat_prompt
    last_bot_message = None
    for ch in chat_history:
        contents += f"{user_prefix}: {ch['user']}\n"
        last_bot_message = ch['bot']
        if last_bot_message is not None:
            contents += f"{bot_prefix}: {last_bot_message}\n"
    user_input = user_input.strip()
    contents += f"{user_prefix}: {user_input}\n{bot_prefix}: "

    prompt, generated = generate(context, contents, True, True)

    stop_index_user = generated.find(f"\n{user_prefix}")
    if stop_index_user < 0:
        stop_index_user = len(generated)
    stop_index_bot = generated.find(f"\n{bot_prefix}")
    if stop_index_bot < 0:
        stop_index_bot = len(generated)
    stop_index = min(stop_index_bot, stop_index_user)
    if stop_index < 0:
        bot_message = generated
    else:
        bot_message = generated[:stop_index].strip()
    bot_message_in_history = bot_message
    if bot_message == last_bot_message:
        bot_message_in_history = None
    chat_history.append({"user": user_input, "bot": bot_message_in_history})
    while len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
    print(f"bot_message={bot_message}")
    return bot_message

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

def qna(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "qna"  
    clear_chat_history(context)
    update.message.reply_text("Q&A 모드로 전환 되었습니다.")

def prompt(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "prompt"  
    clear_chat_history(context)
    update.message.reply_text("prompt 모드로 전환 되었습니다.")

def mqna(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "mqna"  
    clear_chat_history(context)
    update.message.reply_text("다중 Q&A 모드로 전환 되었습니다..")

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
        context.user_data["councelor_type"] = "chatting"
        context.user_data["chat_history"] = []
        update.message.reply_text("기본 채팅 모드입니다. 가능한 명령을 보려면 /help 를 치세요.")
        update.message.reply_text(HELP_TEXT)
        
    q = update.message.text
    q = q.strip()

    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    t = Timer(8, send_typing, [context, update.effective_message.chat_id])  
    t.start()  
    
    a = query(context, q)
    if a is None or len(a) == 0:
        a = "..."
    t.cancel()
    
    update.message.reply_text(a)

def unknown_text(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry I can't recognize you , you said '%s'" % update.message.text)

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(CommandHandler('clear', clear_chat_history_handler))

updater.dispatcher.add_handler(CommandHandler('qna', qna))
updater.dispatcher.add_handler(CommandHandler('prompt', prompt))
updater.dispatcher.add_handler(CommandHandler('chatting', chatting))
updater.dispatcher.add_handler(CommandHandler('therapist', therapist))
updater.dispatcher.add_handler(CommandHandler('doctor', doctor))
updater.dispatcher.add_handler(CommandHandler('mqna', mqna))
updater.dispatcher.add_handler(CommandHandler('mbti', mbti))
updater.dispatcher.add_handler(CommandHandler('testmode', testmode))
updater.dispatcher.add_handler(CommandHandler('normalmode', normalmode))
updater.dispatcher.add_handler(CommandHandler('status', status))

updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown, run_async=True))
updater.dispatcher.add_handler(MessageHandler(
	Filters.command, unknown)) # Filters out unknown commands

# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

updater.start_polling()

print("Ready!")