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

from transformers import AutoTokenizer, logging, pipeline, AutoModelForCausalLM
import torch

chat_prompt = "A는 35세, 성별은 남자이고, 이름은 박길동, 삼성전자 다니는 직장인이다. 애인은 없고, 부모님과 같이 살고 있다. 성격은 친절하고 명랑하다. 묻는 말에 최대한 자세하게 설명해주는 스타일이다.\n아래 대화를 연결해 보시오.\n"
max_output_length = 1024
min_output_length = 256

HELP_TEXT = """
언어모델 챗 봇 by Sempahore.
현재 고물 컴퓨터에서 실행 중이므로 응답 속도가 10초 이상 걸립니다. 

명령어.
/chatting - 일반 잡담 채팅, 35세 직장인 남성을 상대로 하는 채팅임. 사람을 가정하고 하는 채팅. 주제는 제한 없음.
/qna - 질의 응답, 질문에 대해 답을 하며, 이전 질문/답과 연결되지 않음.
/mqna - 다중 질의 응답, 채팅식으로 문답을 이어 나갈 수 있음.
/prompt - 기타 프롬프트 입력

/clear - 채팅 히스토리 삭제
"""

latest_model_dir = "/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-3.8b-multi-func/checkpoint-000"
tokenizer_dir = latest_model_dir
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

updater = Updater(os.environ['TELEGRAM_LM_CHAT'], use_context=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
gpt = AutoModelForCausalLM.from_pretrained(
    latest_model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device, torch.float16)

sep_index = tokenizer.additional_special_tokens.index('<|sep|>')
sep_token_id = tokenizer.additional_special_tokens_ids[sep_index]
print(f'sep_token_id={sep_token_id}')

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
        return chat_query(context, user_input)
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
            
def generate(contents, chat_mode = False):
    contents = contents.strip()
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
    
    output_sequences = gpt.generate(
        encoded_input["input_ids"], 
        do_sample=False,
        early_stopping=True,
        num_beams=3,
        length_penalty=1.0,
        temperature=1.1,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, sep_token_id],
        begin_suppress_tokens=[tokenizer.eos_token_id, sep_token_id],
        # forced_eos_token_id=tokenizer.eos_token_id,
        max_length=max_length
    )
    # print(output_sequences)
    
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
    generated = generated.replace("답변:\n", "").strip()        
    print(f'final generation={generated}')
    print(f'\n\ngarbage={garbage}')        
    
    return prompt, generated
    
def prompt_query(context, user_input):
    content = f"{user_input}"
    prompt, generated = generate(content)
    return generated
        
def qna_query(context, user_input):
    content = f"다음 질문에 답하시오.\n{user_input}"
    prompt, generated = generate(content)
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
    contents += f"질문에 답하시오. 질문이 위 내용과 관련 없으면 위 내용은 무시하시오.\n{user_input}"
    
    prompt, generated = generate(contents)

    bot_message = generated.replace("\n", " ")
    chat_history.append({"user": user_input, "bot": bot_message})
    while len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
    print(f"bot_message={bot_message}")
    return bot_message
        
def chat_query(context, user_input):
    MAX_CHAT_HISTORY = 5
    user_prefix = "A"
    bot_prefix = "B"

    chat_history = context.user_data['chat_history']
    contents = chat_prompt
    for ch in chat_history:
        contents += f"{user_prefix}: {ch['user']}\n{bot_prefix}: {ch['bot']}\n"
    user_input = user_input.strip()
    contents += f"{user_prefix}: {user_input}\n{bot_prefix}: "

    prompt, generated = generate(contents, True)

    stop_index_user = generated.find(f"{user_prefix}:")
    if stop_index_user < 0:
        stop_index_user = len(generated)
    stop_index_bot = generated.find(f"{bot_prefix}:")
    if stop_index_bot < 0:
        stop_index_bot = len(generated)
    stop_index = min(stop_index_bot, stop_index_user)
    if stop_index < 0:
        bot_message = generated
    else:
        bot_message = generated[:stop_index].strip()
    chat_history.append({"user": user_input, "bot": bot_message})
    while len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
    print(f"bot_message={bot_message}")
    return bot_message

def chatting(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "chatting"  
    clear_chat_history(context)
    update.message.reply_text("일반 채팅 모드로 전환 되었습니다.")

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
    update.message.reply_text("mbti 모드로 전환 되었습니다. 먼저 인사를 해보세요.")

    
def clear_chat_history_handler(update: Update, context: CallbackContext):
    clear_chat_history(context)
    update.message.reply_text("채팅 히스토리가 삭제 되었습니다.")

def send_typing(context, chat_id):
    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
def unknown(update: Update, context: CallbackContext):
    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    t = Timer(8, send_typing, [context, update.effective_message.chat_id])  
    t.start()  
    
    if "councelor_type" not in context.user_data.keys():
        context.user_data["councelor_type"] = "chatting"
        context.user_data["chat_history"] = []
        update.message.reply_text("기본 채팅 모드입니다. 가능한 명령을 보려면 /help 를 치세요.")
        update.message.reply_text(HELP_TEXT)
        
    q = update.message.text
    q = q.strip()
    # if not q.endswith("?"):
    #     q = q + "?"
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
updater.dispatcher.add_handler(CommandHandler('mqna', mqna))
updater.dispatcher.add_handler(CommandHandler('mbti', mbti))

updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))
updater.dispatcher.add_handler(MessageHandler(
	Filters.command, unknown)) # Filters out unknown commands

# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

updater.start_polling()

print("Ready!")