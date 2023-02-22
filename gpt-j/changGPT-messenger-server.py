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
from deepspeed.module_inject.containers.gptneox import DS_GPTNEOXContainer, GPTNEOXLayerPolicy
from transformers import GPTNeoXLayer

# facebook messenger section
# tunning facebook web hook to local
# ssh -R 8091:127.0.0.1:5000 lcw.plan4.house
import requests
from flask import Flask, request
from waitress import serve
from multiprocessing import Process
from threading import Thread

from const.prompts import HELP_TEXT, chat_prompt_normal, chat_prompt_therapist, chat_prompt_doctor, chat_prompt_mbti, chat_prompt_expert, chat_prompt_expert2
from const.fortune import job_list, Personality_types, places_to_meet, asian_man_looks, asian_women_looks, wealth
app = Flask(__name__)

fb_veryfy_token = os.environ["FB_VERIFY_TOKEN"]
fb_page_access_token = os.environ["FB_PAGE_ACCESS_TOKEN"]

@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == fb_veryfy_token:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webhook():
    data = request.get_json()
    # log(data)  # {"object": "page", "entry": [{"id": "104026212393785", "time": 1676981468408, "messaging": [{"sender": {"id": "5436183479779412"}, "recipient": {"id": "104026212393785"}, "timestamp": 1676981468032, "message": {"mid": "m_fd7dhzpy9XXc9mnyuqYCl0m_P0t1RoxyDumMyN8ZHUIfL11cMVS5MWPh-ICumR9Xae1Fwc_eBhDdFsFQYFmWFA", "text": "222"}}]}]}
    if data["object"] == "page":
        for entry in data["entry"]:
            chat_id = entry["id"]
            for messaging_event in entry["messaging"]:
                if messaging_event.get("message"):  # someone sent us a message
                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    if "text" not in messaging_event["message"]:
                        print('no text in message.')
                        log(data)
                    else:
                        message_text = messaging_event["message"]["text"]  # the message's text
                        is_echo = False
                        if "is_echo" in messaging_event["message"]:
                            is_echo = messaging_event["message"]["is_echo"]
                        if not is_echo:
                            # fb_handle_user_message(sender_id, message_text, chat_id)
                            #p = Process(target=fb_handle_user_message, args=(sender_id, message_text, chat_id))
                            p = Thread(target=fb_handle_user_message, args=(sender_id, message_text, chat_id))
                            p.start()                        
                            # send_message(sender_id, f'echo {message_text}')
                if messaging_event.get("delivery"):  # delivery confirmation
                    pass
                if messaging_event.get("optin"):  # optin confirmation
                    pass
                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass
    return "ok", 200

def send_message(recipient_id, message_text):
    # log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))
    params = {"access_token": fb_page_access_token}
    headers = {"Content-Type": "application/json"}
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)

def log(msg):  # simple wrapper for logging to stdout on heroku
    if type(msg) is dict:
        msg = json.dumps(msg, ensure_ascii=False)
    else:
        msg = str(msg)
    print(msg)
    
def fb_messenger_start():
    port = 5000
    print(f'running on port {port}')    
    serve(app, host="0.0.0.0", port=port, threads= 8)
    # app.run(debug=True)

    
#
# facebook messenger section end
#
    
latest_model_dir = None
latest_model_dir_on_test = None

max_output_length = 2048
min_output_length = 512
generation_chunk = 20

tokenizer_dir = latest_model_dir
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
deepspeed_mode = False
zero_mode = False
telegram = False
facebook = False
telegram_test_mode = False

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
telegram_test_mode = args.telegram_test_mode

tokenizer_dir = latest_model_dir

if telegram_test_mode:
    print("****** warning: telegram is on test mode.")
    updater = Updater(os.environ['TELEGRAM_LM_CHAT_TEST'], use_context=True)
else:
    updater = Updater(os.environ['TELEGRAM_LM_CHAT'], use_context=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
error_display_token_output = tokenizer('*.*', return_tensors='pt').to(device)['input_ids']

generator = None
gpt = None
gpt_on_test = None
deepspeed_mode = args.deepspeed_mode
zero_mode = args.zero_mode
telegram = args.telegram
facebook = args.facebook

if not zero_mode:
    print(f'normal loading... {latest_model_dir}')
    gpt = AutoModelForCausalLM.from_pretrained(
        latest_model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device, torch.float16)

    if latest_model_dir == latest_model_dir_on_test:
        print("**** normal == test")
        gpt_on_test = gpt
    else:
        print(f'****************test model loading... {latest_model_dir_on_test}')
        gpt_on_test = AutoModelForCausalLM.from_pretrained(
            latest_model_dir_on_test,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            #load_in_8bit=True,
            #device_map='auto',
        ).to(device, torch.float16)
        
if deepspeed_mode:
    print("****************deepspeed_mode enabled!")
    ds_engine = deepspeed.init_inference(
        gpt_on_test,
        tensor_parallel={"enabled":True, "tp_size":1},
        dtype=torch.float16,
        # replace_method='auto',
        checkpoint=None,
        replace_with_kernel_inject=False,
        injection_policy={GPTNeoXLayer: (GPTNEOXLayerPolicy, )}
    )
    gpt_on_test = ds_engine.module
elif zero_mode:
    print("****************zero_mode enabled!")
    generator = mii.mii_query_handle("lcw_deployment")
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)    
    


sep_index = tokenizer.additional_special_tokens.index('<|sep|>')
sep_token_id = tokenizer.additional_special_tokens_ids[sep_index]
tt = tokenizer("\n?.")
newline_token_id = tt['input_ids'][0]
question_mark_token_id = tt['input_ids'][1]
period_token_id = tt['input_ids'][2]
print(f'sep_token_id={sep_token_id}\nnewline_token_id={newline_token_id}\nquestion_mark_token_id={question_mark_token_id}\nperiod_token_id={period_token_id}')
print(tokenizer.decode([224]))

def start(update: Update, context: CallbackContext):
	update.message.reply_text(HELP_TEXT)

def help(update: Update, context: CallbackContext):
	update.message.reply_text(HELP_TEXT)

def clear_chat_history(context):
    context.user_data['chat_history'] = {"normalmode":[], "testmode":[]}
    return

def query(context, message, user_input):
    if context.user_data['councelor_type'] == "chatting":
        return chat_query(context, message, user_input, chat_prompt_normal)
    elif context.user_data['councelor_type'] == "therapist":
        return chat_query(context, message, user_input, chat_prompt_therapist)
    elif context.user_data['councelor_type'] == "doctor":
        return chat_query(context, message, user_input, chat_prompt_doctor)
    elif context.user_data['councelor_type'] == "expert":
        if not user_input.endswith(('?', ".", "!")):
            user_input = user_input + "?"
        return chat_query(context, message, user_input, chat_prompt_expert, "B", "A", 3)
    elif context.user_data['councelor_type'] == "expert2":
        if not user_input.endswith(('?', ".", "!")):
            user_input = user_input + "?"
        return chat_query(context, message, user_input, chat_prompt_expert2, "B", "A", 3)
    elif context.user_data['councelor_type'] == "mbti":
        return chat_query(context, message, user_input, chat_prompt_mbti, "B", "A", 6)
    elif context.user_data['councelor_type'] == "fortune":
        return chat_query(context, message, user_input, context.user_data["prompt"], "B", "A", 2)
    elif context.user_data['councelor_type'] == "prompt":
        return prompt_query(context, message, user_input)
        
def generate_base(model, contents, gen_len):
    encoded_input = tokenizer(contents, return_tensors='pt').to(device)
    print(f"text={len(contents)}, token={encoded_input['input_ids'].size()}")
    input_length = encoded_input['input_ids'].size()[1]
    print(f'input_length={input_length}')
    input_tensor = encoded_input['input_ids']
    for i in range(3):
        try:
            output_sequences = model.generate(
                input_tensor, 
                do_sample=False,
                early_stopping=True,
                use_cache=True,
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
                max_new_tokens=gen_len
            )
        except Exception as e:
            print(f'generate_base error={e}')
            traceback.print_exc()
            output_sequences = torch.cat([input_tensor, error_display_token_output])
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
    return output_text

def generate_base_zero(contents):
    result_id = generator.query_non_block(
        {"query": [contents]}, 
        do_sample=False, 
        max_new_tokens=generation_chunk,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        num_beams=3,       
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
        output = f"{contents}\n음... 뭔가 잘 못 됐어..."
    else:
        output = result
    return output

def search_stop_word(generated):
    stopped = False
    match = re.search(r'\n고객:|\n직원:|\nB는 A|\nA와 B|\nA가\s|<\|endoftext\|>|\n\(|^\(|\n?[A-Z]\s?(?:[:;-]|$)', generated)
    if match is None:
        bot_message = generated
    else:
        stopped = True
        stop_index = match.start()
        bot_message = generated[:stop_index].strip()
        print(f'prefix stop remained = {generated[stop_index:]}')
    return bot_message, stopped

def remove_trash(text):
    text = text.replace("답은 아래와 같습니다.\n", "")        
    text = text.replace("답변:", "")
    text = text.replace("키키", "ㅋㅋ")
    return text
        
def reply_text(context, message, text, full_text, last_sent_msg, flush=False):
    if "facebook" not in context.user_data:
        # print(f'reply_text:full_text=[{full_text}]')
        if flush:
            full_text = full_text.strip() + "◈"
        if last_sent_msg is None:
            last_sent_msg = message.reply_text(full_text)
            print("$$replay_text called.")
        else:
            last_sent_msg.edit_text(full_text)
            print("$$edit_text called.")
        return "", last_sent_msg
    else:
        text = remove_trash(text)
        if flush:
            message.reply_text(text.strip() + "◈")
            print("$$replay_text called. flushed.")
            return None, None
        match = re.search('[\n\.\?\!][\s\n]', text)
        stop_index = -1
        if match is None:
            matches = re.finditer(',\s', text)
            for m in matches:
                if m.start(0) > generation_chunk * 2:
                    stop_index = m.start(0) 
            if stop_index < 0:
                return text, None
        else:
            stop_index = match.start()

        text_to_reply = text[:stop_index+1].strip()
        if len(text_to_reply) > 0:
            message.reply_text(text_to_reply)
            print("$$replay_text called. remained")
        remain_text = text[stop_index+1:]
        # print(f'remain_text=[{remain_text}]')
        return remain_text, None
    
def generate(context, message, contents, open_end = False, gen_len = generation_chunk):
    global generator
    contents = contents.strip()
    if not open_end:
        contents = f'{contents}<|sep|>'
    try:
        testmode = False
        if 'mode' not in context.user_data or context.user_data['mode'] == "normalmode":
            model = gpt
            print(f'running on normal model, {latest_model_dir}.')
        else:
            testmode = True
            model = gpt_on_test
            print(f'running on test model, {latest_model_dir_on_test}.')
        
        gen_text_to_reply = ""
        stopped = False
        gen_text_concat = ""
        generation_count = 0
        sent_message = None
        start_time = datetime.today().timestamp()
        prompt = contents
        print(f'prompt={prompt}')
        while True:
            if generator is not None and zero_mode:
                output = generate_base_zero(contents)
            else:
                output = generate_base(model, contents, gen_len)
            generation_count += 1
            gen_text = output[len(contents):]
            print(f'new generated=[{gen_text}]')
            gen_text, stopped = search_stop_word(gen_text)
            gen_text_concat += gen_text
            gen_text_to_reply += gen_text
            if stopped:
                if len(gen_text) > 0: 
                    print(f'**stop pos={len(gen_text)}')
                    gen_text_to_reply, sent_message = reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message, True)
                break
            gen_text_token = tokenizer(gen_text)['input_ids']
            new_gen_token_len = len(gen_text_token)
            print(f'new_gen_token_len={new_gen_token_len}')
            if new_gen_token_len < generation_chunk or len(gen_text.strip()) == 0 or (new_gen_token_len == generation_chunk and gen_text.strip()[-1:] == "."):
                print(f'**gen shorter than request or end with period ={new_gen_token_len}')
                reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message, True)
                break
            else:
                gen_text_to_reply, sent_message = reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message)
            contents = output         
        print(f'generation_count={generation_count}')
        print(f'gen_text_concat final=[{gen_text_concat}]')
        generated = gen_text_concat

        end_time = datetime.today().timestamp()
        print(f"\n******inference time = {end_time-start_time}")
        generated = remove_trash(generated)
        # print(f'generated={generated}')
    except Exception as e:
        print(f'generate error = {e}')
        traceback.print_exc()
        prompt = "error!"
        generated = "음..."
    
    return prompt, generated
    
def prompt_query(context, message, user_input):
    content = f"{user_input}"
    prompt, generated = generate(context, message, content, True)
    return prompt, generated
        
def chat_query(context, message, user_input, chat_prompt, user_prefix="B", bot_prefix="A", MAX_CHAT_HISTORY=7, CHAT_RESPONSE_LEN=generation_chunk):
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

    prompt, bot_message = generate(context, message, contents, True, CHAT_RESPONSE_LEN)

    bot_message_in_history = bot_message
    if bot_message == last_bot_message:
        bot_message_in_history = None
    timestamp = datetime.today().timestamp()
            
    chat_history.append({"user": user_input, "bot": bot_message_in_history, "time": timestamp})
    while len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
    # print(f"bot_message={bot_message}")
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

def status(message, context: CallbackContext):
    if 'mode' not in context.user_data:
        context.user_data['mode'] = 'normalmode'
    if 'councelor_type' not in context.user_data:
        context.user_data['councelor_type'] = 'expert'
    s = f"runmode = {context.user_data['mode']}\nresponse type={context.user_data['councelor_type']}\nshow normal={context.user_data['shownormal']}"  
    clear_chat_history(context)
    message.reply_text(s)

def clear_chat_history_handler(update: Update, context: CallbackContext):
    clear_chat_history(context)
    update.message.reply_text("채팅 히스토리가 삭제 되었습니다.")

def send_typing(context, chat_id):
    if "facebook" not in context.user_data:
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

def init_user_data(context, clear_history=True):
    if "councelor_type" not in context.user_data:
        context.user_data["councelor_type"] = "expert"
    if "mode" not in context.user_data:
        context.user_data["mode"] = "normalmode"
    if "shownormal" not in context.user_data:
        context.user_data["shownormal"] = False
    if 'chat_history' not in context.user_data:
        context.user_data['chat_history'] = {"normalmode":[], "testmode":[]}
    if clear_history:
        clear_chat_history(context)
            
def unknown(update: Update, context: CallbackContext):
    #print(update)
    if update.message is not None:
        message = update.message
    elif update.edited_message is not None:
        message = update.edited_message
    chat_id = update.effective_message.chat_id
    user_message_handler(message, context, chat_id)
        
def user_message_handler(message, context, chat_id):        
    # print(message)
    """
    {'new_chat_photo': [], 'supergroup_chat_created': False, 'photo': [], 'chat': {'username': 'ninedra9ons', 'id': 858097523, 'type': 'private', 'first_name': 'Chang'}, 'group_chat_created': False, 'new_chat_members': [], 'delete_chat_photo': False, 'entities': [], 'text': '오로라 발생 원인?', 'date': 1676931116, 'channel_chat_created': False, 'message_id': 12617, 'caption_entities': [], 'from': {'id': 858097523, 'username': 'ninedra9ons', 'language_code': 'en', 'first_name': 'Chang', 'is_bot': False}}
    
    """
    username = message.chat['username']
    first_name = message.chat['first_name']
    user_id = message.chat['id']
    
    now = datetime.today()
    q = message.text
    q = q.strip()
    print(f"\n\n---------------\n{now} {first_name}({username}, {user_id}, {chat_id}): {q}\n")

    if q == '--' and 'last_bot_message' in context.user_data:
        message.reply_text(context.user_data['last_bot_message'])
        return
        
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
            status(message, context)

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
                    
    context.user_data['chat_id'] = chat_id
    send_typing(context, chat_id)
    
    q_start_time = datetime.today()
    q_start_time_str = q_start_time.strftime('%Y.%m.%d %H:%M:%S')
    prompt, a = query(context, message, q)
    q_end_time = datetime.today()
    q_end_time_str = q_end_time.strftime('%Y.%m.%d %H:%M:%S')
    duration = q_end_time - q_start_time 
    
    a = a.strip()
    print(f'{q_start_time_str} Q:{q}\n{q_end_time_str} A:{a}\n{duration}\n\n')

    if "shownormal" not in context.user_data.keys():
        context.user_data['shownormal'] = False 
    show_normal = context.user_data["shownormal"]
    
    if len(a) == 0 or prompt=='error!':
        a = "음..."
        clear_chat_history(context)
        print('no generation, retry with clear chat history.')
        message.reply_text("잠깐만... 오류났다...")
        prompt, a = query(context, message, q)
        a = a.strip()
    else:
        if show_normal:
            message.reply_text(a)
        context.user_data['last_bot_message'] = a
        data_path = f"{os.path.expanduser('~')}/Documents/changGPT-chatdata"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_name_for_chat = f'{chat_id}-{user_id}-{first_name}.txt'
        path = os.path.join(data_path, file_name_for_chat)
        with open(path, "a") as myfile:
            myfile.write(f'{q_start_time_str} Q:{q}\n{q_end_time_str} A:{a}\n{duration}\n\n')
        
    if "mode" not in context.user_data.keys():
        context.user_data['mode'] = "normalmode" 
    if context.user_data['mode'] == "testmode" and show_normal:
        context.user_data['mode'] = "normalmode"
        prompt, a2 = query(context, message, q)
        a2 = a2.strip()
        context.user_data['mode'] = "testmode"
        message.reply_text(f'{a2}-[N]')
        
def unknown_text(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry I can't recognize you , you said '%s'" % update.message.text)

#
#   facebook section
#
class ContextFB:
    def __init__(self):
        self.user_data = {}
 
class MessageFB:
    def __init__(self, context, text):
        self.context = context
        self.text = text
        self.chat = {}
        
    def reply_text(self, text):
        user_id = self.context.user_data['user_id']
        send_message(user_id, text)
    
contexts_dict = {}
def fb_handle_user_message(user_id, text, chat_id):
    global contexts_dict
    
    print(f'facebook message = [{text}]')
    if user_id in contexts_dict:
        context = contexts_dict[user_id]
    else:
        context = ContextFB()
        contexts_dict[user_id] = context
        
    context.user_data['user_id'] = user_id
    context.user_data['facebook'] = True
    init_user_data(context, False)
    message = MessageFB(context, text)
    message.chat['username'] = user_id
    message.chat['first_name'] = user_id
    message.chat['id'] = user_id
    user_message_handler(message, context, chat_id)

# fb_handle_user_message('aaa', "안녕", '3333')

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

if telegram:
    print("starting telegram bot...")
    updater.start_polling()

if facebook:
    print("starting facebook bot...")
    fb_messenger_start()    # should be last one