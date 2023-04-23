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
from telegram.bot import Bot, BotCommand

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
import asyncio

from const.prompts import HELP_TEXT, chat_prompt_normal, chat_prompt_therapist, chat_prompt_doctor, \
            chat_prompt_mbti, chat_prompt_expert_ko, chat_prompt_expert_en, chat_prompt_expert2, article_writing, \
            blog_writing, receipe_writing, poem_writing, today_fortune_writing, today_fortune_keyword, \
            entity_extract_for_poem, samhangsi_writing, movie_info, detail_answer_prompt, detail_answer_prompt_fortune, \
            entity_extract_name, entity_extract_name_for_samhangsi, prompt_saju_consulting, prompt_dirty_OPT, \
            chat_prompt_expert_test_mode
from const.fortune import job_list, Personality_types, places_to_meet, asian_man_looks, asian_women_looks, wealth

from plugin.todays_fortune import get_todays_fortune
import openai
from text_generation import Client

app = Flask(__name__)

fb_veryfy_token = os.environ["FB_VERIFY_TOKEN"]
fb_page_access_token = os.environ["FB_PAGE_ACCESS_TOKEN"]
openai.api_key = os.environ["OPENAI_API_KEY"]

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
openai_api_base = None
basaran_api_base = "http://127.0.0.1:8888/v1"
hf_tgi_api_base = "http://127.0.0.1:8080"

max_output_length = 2048
min_output_length = 512
generation_chunk = 48

tokenizer_dir = latest_model_dir
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
deepspeed_mode = False
zero_mode = False
basaran_mode = False
hf_tgi_mode = False
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

deepspeed_mode = args.deepspeed_mode
zero_mode = args.zero_mode
basaran_mode = args.basaran_mode
hf_tgi_mode = args.hf_tgi_mode
telegram = args.telegram
facebook = args.facebook
hf_tgi_api_base = args.hf_tgi_api_base
basaran_api_base = args.basaran_api_base
streaming = args.streaming

latest_model_dir = os.environ['CURRENT_MODEL']
latest_model_dir_on_test = os.environ['CURRENT_MODEL']
telegram_test_mode = args.telegram_test_mode

tokenizer_dir = latest_model_dir

rasa_agent = None
if args.rasa_agent:
    from rasa.core.agent import Agent
    rasa_agent = Agent.load(model_path='./rasa/models/model.tar.gz')

if telegram_test_mode:
    print("****** warning: telegram is on test mode.")
    updater = Updater(os.environ['TELEGRAM_LM_CHAT_TEST'], use_context=True)
else:
    updater = Updater(os.environ['TELEGRAM_LM_CHAT'], use_context=True)

generator = None
generator_on_test = None
gpt = None
gpt_on_test = None

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

if not (zero_mode or basaran_mode or hf_tgi_mode):
    error_display_token_output = tokenizer('*.*', return_tensors='pt').to(device)['input_ids']
    sep_index = tokenizer.additional_special_tokens.index('<|sep|>')
    sep_token_id = tokenizer.additional_special_tokens_ids[sep_index]
    tt = tokenizer("\n?.")
    newline_token_id = tt['input_ids'][0]
    question_mark_token_id = tt['input_ids'][1]
    period_token_id = tt['input_ids'][2]
    print(f'sep_token_id={sep_token_id}\nnewline_token_id={newline_token_id}\nquestion_mark_token_id={question_mark_token_id}\nperiod_token_id={period_token_id}')
    # print(tokenizer.decode([224]))

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
        gpt,
        tensor_parallel={"enabled":True, "tp_size":1},
        dtype=torch.float16,
        # replace_method='auto',
        checkpoint=None,
        replace_with_kernel_inject=False,
        injection_policy={GPTNeoXLayer: (GPTNEOXLayerPolicy, )}
    )
    gpt = ds_engine.module
elif zero_mode:
    print("****************zero_mode enabled!")
    generator = mii.mii_query_handle("lcw_deployment")
    try:
        print("****************loading lcw_deployment_test.")
        generator_on_test = mii.mii_query_handle("lcw_deployment_test")
    except:
        print("zeromode: no generator on test")
        generator_on_test = generator
elif basaran_mode:
    openai_api_base = openai.api_base
    openai.api_base = basaran_api_base

asyncio_loop = asyncio.get_event_loop()

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
        if telegram_test_mode:
            return chat_query(context, message, user_input, chat_prompt_expert_test_mode, "B", "A", 12)
        if 'language' not in context.user_data:
            context.user_data['language'] = 'ko'
        if context.user_data['language'] == 'ko':
            return chat_query(context, message, user_input, chat_prompt_expert_ko, "B", "A", 12)
        elif context.user_data['language'] == 'en':
            return chat_query(context, message, user_input, chat_prompt_expert_en, "B", "A", 5)
    elif context.user_data['councelor_type'] == "expert2":
        return chat_query(context, message, user_input, chat_prompt_expert2, "B", "A", 5)
    elif context.user_data['councelor_type'] == "mbti":
        return chat_query(context, message, user_input, chat_prompt_mbti, "B", "A", 6)
    elif context.user_data['councelor_type'] == "saju":
        return chat_query(context, message, user_input, prompt_saju_consulting, "B", "A", 6)
    elif context.user_data['councelor_type'] == "dirty":
        return chat_query(context, message, user_input, prompt_dirty_OPT, "B", "Emma", 12)
    elif context.user_data['councelor_type'] == "fortune":
        return chat_query(context, message, user_input, context.user_data["prompt"], "B", "A", 2)
    elif context.user_data['councelor_type'] == "prompt":
        return prompt_query(context, message, user_input)

generation_kwargs_beam1 = {
    "do_sample":False,
    "early_stopping":False,
    "use_cache":True,
    "num_beams":4,
    # "length_penalty":5.0,
    "temperature":0.4,
    # "top_k":4,
    # "top_p":0.6,
    "no_repeat_ngram_size":2, # if change to 3, normally very short generation
    "repetition_penalty":1.2,
    # "pad_token_id":tokenizer.eos_token_id,
}
        
generation_kwargs_beam = {
    "do_sample":False,
    "early_stopping":False,
    "use_cache":True,
    "num_beams":4,
    "length_penalty":5.0,
    "temperature":0.4,
    # "top_k":4,
    # "top_p":0.6,
    "no_repeat_ngram_size":3, 
    # "repetition_penalty":0.7,
    # "pad_token_id":tokenizer.eos_token_id,
}

generation_kwargs_contrasive = {
    "do_sample":True,
    "early_stopping":False,
    "use_cache":False,
    # "length_penalty":0.1,
    # "num_beams":3,
    # "length_penalty":1.0,
    "temperature":0.5,
    "penalty_alpha":0.6,     
    "top_k":40,
    # "top_p":0.4,
    "no_repeat_ngram_size":3,       
    "repetition_penalty":1.2,
    # "pad_token_id":tokenizer.eos_token_id,
}

generation_kwargs_basaran = {
    # "do_sample":False,
    # "use_cache":False,
    # "early_stopping":True,
    # "length_penalty":9.0,
    "temperature":0.7,
    # "top_k":40,
    "top_p":0.95,
    # "no_repeat_ngram_size":2, 
    # "repetition_penalty":50.0,
    # "pad_token_id":tokenizer.eos_token_id,
}

generation_kwargs_basaran_test_opt = {
    "do_sample":False,
    "use_cache":False,
    "early_stopping":True,
    # "length_penalty":10.0,
    "temperature":1.0,
    # "top_k":40,
    "top_p":0.90,
    "no_repeat_ngram_size":20, 
    "repetition_penalty":50.0,
    # "pad_token_id":tokenizer.eos_token_id,
}

generation_kwargs_hf_tgi = {
    "do_sample": False,
    # "repetition_penalty": 1.1,
    "return_full_text": False,
    "seed": None,
    "stop_sequences": [
    ],
    "temperature": 0.7,
    # "top_k": 10,
    "top_p": 0.8,
    "truncate": None,
    "typical_p": 0.9,
    "watermark": False
}

if basaran_mode:
    generation_kwargs = generation_kwargs_basaran
elif hf_tgi_mode:
    generation_kwargs = generation_kwargs_hf_tgi
else:
    generation_kwargs = generation_kwargs_beam1

def generate_base(model, contents, gen_len):
    encoded_input = tokenizer(contents, return_tensors='pt').to(device)
    print(f"text={len(contents)}, token={encoded_input['input_ids'].size()}")
    input_length = encoded_input['input_ids'].size()[1]
    print(f'input_length={input_length}')
    input_tensor = encoded_input['input_ids']
    generation_kwargs["max_new_tokens"] = gen_len
    for i in range(3):
        try:
            output_sequences = model.generate(
                input_tensor, 
                eos_token_id=[tokenizer.eos_token_id, sep_token_id],
                begin_suppress_tokens=[tokenizer.eos_token_id, sep_token_id, newline_token_id, question_mark_token_id, period_token_id],
                **generation_kwargs
            )
        except Exception as e:
            print(f'generate_base error={e}')
            traceback.print_exc()
            output_sequences = torch.cat([input_tensor, error_display_token_output])
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
    return output_text

def generate_base_zero(zero_generator, contents, gen_len = generation_chunk):
    generation_kwargs["max_new_tokens"] = gen_len
    result_id = zero_generator.query_non_block(
        {"query": [contents]}, 
        eos_token_id=tokenizer.eos_token_id,
        **generation_kwargs
    )
    result = None
    for i in range(100):
        result = zero_generator.get_pending_task_result(result_id)
        if result is not None:
            result = result.response[0]
            break
        time.sleep(2)
    if result is None:
        output = f"{contents}\n음... 뭔가 잘 못 됐어..."
    else:
        output = result
    return output

def search_stop_word(generated):
    stopped = False
    match = re.search(r'<\|endoftext\|>|\n\n\n\n|</s>|\|sep\|>|\n#|\nB$|\n고객:|\n직원:|\nB는 A|\nA와 B|\nA가\s|\n[A-Z]\s?[\.:;-]', generated)
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
            text = text.strip() + "◈"
        remain_text = text
        if last_sent_msg is None:
            last_sent_msg = message.reply_text(text)
            print("$$reply_text called.")
        else:
            #print(f"$$edit_text called=[{full_text}]")
            try:
                last_sent_msg.edit_text(text)
                if len(text) > 3000:
                    last_sent_msg = None
                    remain_text = ""
            except Exception as e:
                print(f"reply_text exception = {e}")
        return remain_text, last_sent_msg
    else:
        text = remove_trash(text)
        if flush:
            message.reply_text(text.strip() + "◈")
            print("$$reply_text called. flushed.")
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
        remain_text = text[stop_index+1:]
        print(f'**reply text, remain_text=[{remain_text}]')
        return remain_text, None
    
def generate_low_level(context, contents, gen_len = generation_chunk, add_eos = True):
    contents = contents.strip()
    if 'chatgpt' in context.user_data:
        chatgpt_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": contents}
            ]
        )
        msg = json.dumps(chatgpt_output, ensure_ascii=False)
        out = chatgpt_output['choices'][0]['message']['content']
        # print(f'---chatgpt---in={len(contents)},out={len(out)}\n{msg}')

        output = contents + out
        if add_eos:
            output += '<|endoftext|>'
        return output
    elif basaran_mode:
        response = openai.Completion.create(
            model='chang',
            prompt=contents,
            max_tokens=gen_len,
            stream=False,  # this time, we set stream=True
            **generation_kwargs,
        )
        msg = json.dumps(response, ensure_ascii=False)
        # print(f'---basaran---out={msg}')
        out = response['choices'][0]['text']
        
        output = contents + out
        if add_eos:
            output += '<|endoftext|>'
        return output
    elif hf_tgi_mode:
        client = Client(hf_tgi_api_base)
        response = client.generate(contents, max_new_tokens=gen_len, **generation_kwargs)
        # print(f'---hf-tgi---out={response}')
        out = response.generated_text
        output = contents + out
        if add_eos:
            output += '<|endoftext|>'
        return output

    if 'mode' not in context.user_data or context.user_data['mode'] == "normalmode":
        model = gpt
        zero_generator = generator 
        print(f'running on normal model')
    else:
        model = gpt_on_test
        zero_generator = generator_on_test 
        print(f'running on test model')
    if generator is not None and zero_mode:
        output = generate_base_zero(zero_generator, contents, gen_len)
    else:
        output = generate_base(model, contents, gen_len)
    return output

def generate_and_stop(context, contents, gen_len = generation_chunk):
    #print(f"generate_and_stop input=[{contents}]")
    output = generate_low_level(context, contents, gen_len)
    #print(f"generate_and_stop output=[{output}]")
    output = output[len(contents):].strip()
    output, stopped = search_stop_word(output)
    return output
    
def generate(context, message, contents, open_end = False, gen_len = generation_chunk, max_new_token = 200):
    global generator
    contents = contents.strip()
    if not open_end:
        contents = f'{contents}<|sep|>'
    try:
        gen_text_to_reply = ""
        stopped = False
        gen_text_concat = ""
        generation_count = 0
        sent_message = None
        start_time = datetime.today().timestamp()
        prompt = contents
        print(f'prompt={prompt}')
        context.user_data.pop('stop_generation', None)
        # if basaran_mode or (hf_tgi_mode and not telegram_test_mode) or 'chatgpt' in context.user_data:
        if streaming or 'chatgpt' in context.user_data:
            speed = 0.1 #smaller is faster
            max_response_length = 1500
            start_time = time.time()
            # Generate Answer
            kwargs = generation_kwargs
            if 'chatgpt' in context.user_data:
                kwargs = {}
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    stream=True,
                    messages=[
                        {"role": "user", "content": contents}
                    ]
                )
                # msg = json.dumps(chatgpt_output, ensure_ascii=False)
                # out = chatgpt_output['choices'][0]['message']['content']
            elif basaran_mode:
                response = openai.Completion.create(
                    model='text-davinci-003',
                    prompt=prompt,
                    max_tokens=max_response_length,
                    stream=True,  # this time, we set stream=True
                    **kwargs,
                )
            elif hf_tgi_mode:
                client = Client(hf_tgi_api_base)
                response = client.generate_stream(contents, max_new_tokens=max_new_token, **kwargs)

            # Stream Answer
            temp_gen_text_concat = ""
            temp_gen_text_concat_start_pos = 0
            no_gen_count = 0
            stopped = False
            for event in response:
                event_time = time.time() - start_time  # calculate the time delay of the event
                if 'chatgpt' in context.user_data:
                    # print(event)
                    gen_text = ""
                    d0 = event['choices'][0]
                    if 'content' in d0['delta']:
                        gen_text = d0['delta']['content']
                    if 'finish_reason' in d0 and d0['finish_reason'] == "stop":
                        stopped = True
                elif basaran_mode:
                    gen_text = event['choices'][0]['text']  # extract the text
                elif hf_tgi_mode:
                    id_str = tokenizer.decode([event.token.id])
                    print(f"'{event.token.text}',", end="")
                    # if not event.token.special:
                    gen_text = event.token.text
                else:
                    print(f"mode error----------------")
                    break
                # if len(gen_text) > 0:
                #     print(f"finish_reason = {event['choices'][0]['finish_reason']}, {gen_text}, {ord(gen_text[0])}")
                time.sleep(speed)
                if len(gen_text) == 0 and not stopped:
                    no_gen_count += 1
                    print(f"no gen text={no_gen_count}")
                    if no_gen_count > 5:
                        stopped = True
                    else:
                        continue
                no_gen_count = 0
                prev_len = len(gen_text_concat)
                gen_text_concat += gen_text
                if not stopped:
                    gen_text_concat, stopped = search_stop_word(gen_text_concat)
                gen_text = gen_text_concat[prev_len:]
                if len(gen_text) > 0 and not stopped:
                    temp_gen_text_concat += gen_text
                    if len(temp_gen_text_concat) < generation_chunk:
                        continue
                    gen_text_to_reply += temp_gen_text_concat
                    temp_gen_text_concat_start_pos += len(temp_gen_text_concat)
                    print(f"[{temp_gen_text_concat}]={temp_gen_text_concat_start_pos}")
                    temp_gen_text_concat = ""
                    gen_text_to_reply, sent_message = reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message)
                if 'stop_generation' in context.user_data:
                    print('stop_generation detected...')
                    context.user_data.pop('stop_generation', None)
                    stopped = True
                if stopped:
                    print(f"{len(gen_text_concat)=}, {temp_gen_text_concat_start_pos=}")
                    # stop_pos = len(gen_text_concat) - temp_gen_text_concat_start_pos + 1
                    # if stop_pos < 0:
                    #     stop_pos = len(gen_text_concat)
                    # temp_gen_text_concat = temp_gen_text_concat[:stop_pos]
                    gen_text_to_reply = gen_text_concat
                    reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message, True)
                    break
        else:
            while True:
                send_typing(context, context.user_data['chat_id'])
                output = generate_low_level(context, contents, gen_len, False)
                generation_count += 1
                gen_text = output[len(contents):]
                print(f'new generated=[{gen_text}]')
                force_continue = False
                if gen_text.endswith('�'):
                    output = output[:-1]
                    gen_text = gen_text[:-1]
                    if gen_text.endswith('�'):
                        output = output[:-1]
                        gen_text = gen_text[:-1]
                    force_continue = True
                prev_len = len(gen_text_concat)
                gen_text_concat += gen_text
                gen_text_concat, stopped = search_stop_word(gen_text_concat)
                gen_text = gen_text_concat[prev_len:]
                gen_text_to_reply += gen_text
                gen_text_token = tokenizer(gen_text)['input_ids'][:generation_chunk]
                new_gen_token_len = len(gen_text_token)
                print(f'new_gen_token_len={new_gen_token_len}')
                if 'stop_generation' in context.user_data:
                    print('stop_generation detected...')
                    context.user_data.pop('stop_generation', None)
                    stopped = True
                if not force_continue and (stopped or new_gen_token_len < generation_chunk or len(gen_text.strip()) == 0):
                    print(f'**stop pos={len(gen_text)}, new_gen_token_len={new_gen_token_len}, stopped={stopped}')
                    if not stopped and new_gen_token_len >= generation_chunk - 3:
                        print("**** 3 token small case, do not stop!")
                        pass
                    else:
                        reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message, True)
                        break
                gen_text_to_reply, sent_message = reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message)
                contents = output         

        print(f'generation_count={generation_count}')
        print(f'gen_text_concat final=[{gen_text_concat}]')
        generated = gen_text_concat.strip()

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
    prompt, generated = generate(context, message, content, True, 32, 1000)
    return prompt, generated

def build_chat_prompt(chat_history, chat_prompt, user_input, user_prefix, bot_prefix):
    contents = chat_prompt
    last_bot_message = None
    # duplicated = next((item for item in chat_history if item["user"] == user_input), None)
    # if duplicated is not None:
    #     chat_history.remove(duplicated)
    for ch in chat_history:
        contents += f"{user_prefix}: {ch['user']}\n"
        last_bot_message = ch['bot']
        if last_bot_message is not None:
            contents += f'{bot_prefix}: {last_bot_message}\n'
    return last_bot_message, contents

def start_ask_birthday(context):
    if context.user_data["councelor_type"] == 'expert':
        reply = "생년월일시를 입력 하세요. 시는 모르면 입력 안해도 됩니다. 양식은 1988.12.12 13:12, 1999/12/13 18:12 등입니다."
    else:
        reply = "생년월일시를 입력 해. 시는 모르면 입력 안해도 돼. 양식은 1988.12.12 13:12, 1999/12/13 18:12 등으로 하면 돼."
    context.user_data['fortune_data_input_state'] = 'wait_for_birthday'
    return reply

init_bot_message = """
안녕하세요. 반갑습니다. 
저는 딥러닝 기술로 개발된 언어모델입니다. 58억개의 파라메터를 가지고 있으며 요즘 유행하는 대형 언어 모델 보다는 한참 작습니다. 
크기는 작지만 한글에 특화된 학습을 받았습니다. 다소 지식이 부족하지만 언어모델이 대부분 그러하듯 적절한 답변을 해 드릴 수 있습니다. 
저는 인간이 아니기 때문에 직업이나 성별, 취미등 인간이 가진 개인 신상정보는 없습니다. 물어 보셔도 답을 드릴 방법이 없습니다. 
이름은 '창 GPT'를 쓰고 있습니다. 만든 사람의 이름을 빌렸습니다.
"""    
    
def parse_special_input(context, message, user_input):
    user_input = re.sub(r"[\?\.]$", "", user_input.strip())
    result = asyncio.run(rasa_agent.parse_message(message_data=user_input))
    print(result)
    intent = result['intent']
    confidence = intent['confidence']
    intent_name = intent['name']
    print(f"intent={intent_name}, confidence={confidence}")
    contents = None
    reply = None
    do_not_send_reply = False
    if confidence < 0.98:
        return None, None, None, do_not_send_reply
    if False:
        pass
    # elif intent_name == "ask_article":
    #     contents = f"{article_writing}제목: {user_input}\n기사:"
    # elif intent_name == "ask_blog":
    #     contents = f"{blog_writing}제목: {user_input}\n블로그:"
    # elif intent_name == "request_receipe":
    #     contents = f"{receipe_writing}요리 이름: {user_input}\n만드는 법:"
    # elif intent_name == "movie_info":
    #     contents = f"{movie_info}{user_input}"
    # elif intent_name == "request_poem":
    #     content = f'{entity_extract_for_poem}{user_input} ==>'
    #     title = generate_and_stop(context, content)
    #     contents = f"{poem_writing}제목: {title}\n시:"
    elif intent_name == "request_samhangsi":
        #content = f'{entity_extract_name_for_samhangsi}{user_input} =>'
        #name = generate_and_stop(context, content)
        match = re.search(r'^\S+', user_input)
        name = match.group(0)
        content = f"{samhangsi_writing}{name}\n{name[:1]}:"
        print(content)
        samhangsi = generate_and_stop(context, content, 80)
        reply = f"{name[:1]}:{name[:1]}{samhangsi}"
        contents = None
    # elif intent_name == "movie_recommend":
    #     content = f'{entity_extract_for_poem}{user_input} ==>'
    #     movie_title = generate_and_stop(context, content)
    #     content = f"영화 추천목록\n• {movie_title}"
    #     gen_text_concat = ""
    #     sent_message = None
    #     num_recommend = 7
    #     for i in range(num_recommend):
    #         content += "\n•"
    #         print(content)
    #         recommend = generate_low_level(context, content, 50)[len(content):]
    #         recommend = recommend.strip()
    #         print(recommend)
    #         match = re.search(r'•', recommend)
    #         if match is not None:
    #             recommend = recommend[:match.start()].strip()
    #         gen_text_concat += f"• {recommend}\n"
    #         _, sent_message = reply_text(context, message, None, gen_text_concat, sent_message, i == num_recommend - 1)
    #         # message.reply_text(f"*. {recommend}")
    #         content += recommend
    #     contents = None
    #     reply = gen_text_concat
    #     do_not_send_reply = True
    # elif intent_name == "english_mode":
    #     if context.user_data['language'] != "en":
    #         context.user_data['language'] = "en"
    #         clear_chat_history(context)
    #         contents = None
    #         reply = "From now on, we will speak in English."
    # elif intent_name == "korean_mode":
    #     if context.user_data['language'] != "ko":
    #         context.user_data['language'] = "ko"
    #         clear_chat_history(context)
    #         contents = None
    #         reply = "지금 부터는 한국말로 이야기 합니다."
    elif intent_name == "today_fortune" and user_input.startswith("오늘의 운세"):
        contents = None
        reply = start_ask_birthday(context)
    elif intent_name == "greeting":
        contents = None
        if context.user_data["councelor_type"] == 'expert':
            reply = init_bot_message
        else:
            reply = "안녕? 반갑다. 뭐든 물어봐."
        
    return contents, reply, intent_name, do_not_send_reply

def stop_fortune_mode(context, message):
    context.user_data.pop('fortune_data_input_state', None)
    context.user_data.pop('birthday', None)
    context.user_data.pop('sex', None)
    bot_message = "운세를 중단합니다."
    message.reply_text(bot_message)
    return bot_message
    
def handle_story(context, message, contents, user_input):
    bot_message = None
    if 'fortune_data_input_state' not in context.user_data:
        return contents, bot_message
    
    if 'wait_for_birthday' == context.user_data['fortune_data_input_state']:
        match = re.search(r'^[1-9]', user_input)
        if match is not None:
            birtyday = parse_date_input(user_input)
            if birtyday is not None:
                print(f'birthday input successful={birtyday}')
                context.user_data["birthday"] = birtyday
                birthday_str = context.user_data["birthday"].strftime('%Y년 %m월 %d일 %I시 %M분 %p')
                bot_message = f"생시가 {birthday_str}, 맞나 확인해 주세요. 맞아, 틀려 외의 문장을 입력 하면 운세 진행이 중단됩니다."
                message.reply_text(bot_message)
                context.user_data['fortune_data_input_state'] = 'wait_for_confirm'
            else:
                bot_message = "생일을 입력 해야 합니다. 양식이 맞지 않습니다.. 날짜 형식이 아닌 일반 문장을 입력 하면 운세 진행이 중단됩니다."
                message.reply_text(bot_message)
                message.reply_text("생년월일과 출생시간을 입력 해야 합니다. 1980년 3월 20일 오후 2시 20분 또는 1999.2.12 22:00, 1988/12/31 오후 1:30, 198003200220 같은 형식으로 하면 됩니다.")
        else:
            if user_input.startswith('--'):
                keystr = user_input[2:]
                contents = f"{today_fortune_writing}운세 키워드: {keystr}\n오늘의 운세:"
                context.user_data.pop('fortune_data_input_state', None)
            else:
                bot_message = stop_fortune_mode(context, message)
    elif 'wait_for_confirm' == context.user_data['fortune_data_input_state']:
        c, r, i, _ = parse_special_input(context, message, user_input)
        if i == 'confirm':
            context.user_data['fortune_data_input_state'] = 'wait_for_sex'
            bot_message = "성별을 입력 해 주세요. 성별외의 단어를 입력 하면 운세 진행이 중단 됩니다."
            message.reply_text(bot_message)
        elif i == 'deny':
            bot_message = start_ask_birthday(context)
            message.reply_text(bot_message)
        else:
            bot_message = stop_fortune_mode(context, message)
    elif 'wait_for_sex' == context.user_data['fortune_data_input_state']:
        c, r, i, _ = parse_special_input(context, message, user_input)
        if i == 'state_male':
            context.user_data['sex'] = 'male'
        elif i == 'state_female':
            context.user_data['sex'] = 'female'
        else:
            bot_message = stop_fortune_mode(context, message)
        if 'birthday' in context.user_data and 'sex' in context.user_data:
            birthday_str = context.user_data["birthday"].strftime('%Y%m%d%H%M')
            today_str = datetime.today().strftime('%Y%m%d%H%M')
            saju, target_date_samju, fortune = get_todays_fortune(context.user_data['sex'], birthday_str, today_str)
            message.reply_text(saju)
            message.reply_text(target_date_samju)
            print(fortune)
            fortune = re.sub(r"[\.\?\!]", "", fortune)
            ff = fortune.split(' ')
            len_ff = len(ff)
            sel_len = 8
            pos_sel = datetime.today().day % (len_ff - sel_len)
            fortune_keyword = ' '.join(ff[pos_sel:pos_sel+sel_len])
            contents = f"{today_fortune_writing}운세 키워드: {fortune_keyword}\n오늘의 운세:"
        context.user_data.pop('fortune_data_input_state', None)
        context.user_data.pop('birthday', None)
        context.user_data.pop('sex', None)
    return contents, bot_message

def chat_query(context, message, user_input, chat_prompt, user_prefix="B", bot_prefix="A", MAX_CHAT_HISTORY=7, CHAT_RESPONSE_LEN=generation_chunk):
    chat_history = context.user_data['chat_history'][context.user_data['mode']]
    # if len(chat_history) == 0:
    #     chat_history.append({"user": "너의 정체를 밝혀라.", "bot": init_bot_message.strip(), "time": datetime.today().timestamp()})
    last_bot_message, contents = build_chat_prompt(chat_history, chat_prompt, user_input, user_prefix, bot_prefix)
    user_input = user_input.strip()
    bot_message = None

    prompt = ""
    contents += f"{user_prefix}: {user_input}\n{bot_prefix}:"
        
    contents, bot_message = handle_story(context, message, contents, user_input)
    if bot_message is None:
        if rasa_agent is not None:
            c, r, _, do_not_reply = parse_special_input(context, message, user_input)
            if c is not None:
                contents = c
            elif r is not None:
                bot_message = r
                prompt = contents
                if not do_not_reply:
                    reply_text(context, message, bot_message, bot_message, None, True)
        if bot_message is None:
            prompt, bot_message = generate(context, message, contents, True, CHAT_RESPONSE_LEN, 900)

    bot_message_in_history = bot_message
    if bot_message == last_bot_message:
        bot_message_in_history = None
    timestamp = datetime.today().timestamp()
            
    if prompt == "error!":
        return prompt, bot_message
        
    chat_history.append({"user": user_input, "bot": bot_message_in_history, "time": timestamp})
    _, contents = build_chat_prompt(chat_history, chat_prompt, None, user_prefix, bot_prefix)
    #while len(chat_history) > MAX_CHAT_HISTORY:
    tokens = tokenizer(contents)['input_ids']
    print(f'len(tokens) = {len(tokens)}, len(text) = {len(contents)}')
    while len(tokens) > 1024 or len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
        _, contents = build_chat_prompt(chat_history, chat_prompt, None, user_prefix, bot_prefix)
        tokens = tokenizer(contents)['input_ids']
        print(f'len(tokens) = {len(tokens)}, len(text) = {len(contents)}')
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
        sex_partner_str2 = "그남자"
    appearance = f"키는 {looks[appearance_index]['height']}센티미터 이고, 몸무게는 {looks[appearance_index]['weight']} 이며, {looks[appearance_index]['appearance']} 정도로 보여"
    money_index = int(birtyday.day * birtyday.hour) % 12
    money = f"{wealth[money_index]['properties']} 정도로 예상된다."
    fortune_prompt1 = f"""
B는 사주를 통해 연애상담을 하는 고객이다.
B는 {sex_str}인데 앞으로 만날 {sex_partner_str}에 대해서 궁금해서 온라인 채팅을 통해서 연애 상담중이다.
B, 즉 고객을 지칭할떄는 "너"라고 하면 된다.
이후 질문에 대한 모든 답변은 완전 반말체로 한다. 
{sex_partner_str2}는 고객이 만날 수도 있는 미래의 연인이다.
{sex_partner_str2}의 정보는 아래와 같다.
{sex_partner_str2}는 싱글이다.
{sex_partner_str2}의 성별은 {sex_partner_str} 이다.
{sex_partner_str2}의 직업은 {job} 중 하나일 가능성이 높다.
{sex_partner_str2}의 성격은 {personality} 일 가능성이 있어.
{sex_partner_str2}을 만나는 시기는 {meet_when}.
{sex_partner_str2}을 만나는 장소는 {meet_where}.
{sex_partner_str2}의 외모는 {appearance}.
{sex_partner_str2}의 재산은 {money}.
{sex_partner_str2}의 이름은 당연히 알 수가 없어.
위 내용에 기반하여 답변을 하되, 내용에 없는 질문에 대해서는 "상담 범위를 벗어난다"는 취지로 답변한다. 대화는 연애상담에 국한한다.
위에서 제공되지 않은 정보에 대한 질문에 대해서는 {sex_partner_str2}의 생일을 모르기 때문에 명리학 적으로 정확한 예측을 할 수 없다고 답변해. 
B: 지금부터 모든 답변은 완전 반말로 한다. 알았지?
A: 알았어.
B: 오늘날씨?
A: 그건 상담범위를 벗어나는 질문이야. 연애상담에 집중 해 줄래?
"""
    fortune_prompt2 = f"""
B는 사주를 통해 연애상담을 하는 고객이다.
B는 {sex_str}인데 앞으로 만날 {sex_partner_str}에 대해서 궁금해서 온라인 채팅을 통해서 연애 상담중이다.
B: 그녀의 성격은 어때?
A: {sex_partner_str2}의 성격은 {personality} 일 가능성이 있어.
B: 그녀의 성별은?
A: {sex_partner_str2}의 성별은 {sex_partner_str} 이다.
B: 그녀의 직업은?
A: {sex_partner_str2}의 직업은 {job} 중 하나일 가능성이 높다.
B: 오늘날씨?
A: 그건 상담범위를 벗어나는 질문이야. 연애상담에 집중 해 줄래?
B: 그녀을 언제 만나게 돼?
A: {sex_partner_str2}을 만나는 시기는 {meet_when}.
B: 어디서 그녀을 만나게 되지?
A: {sex_partner_str2}을 만나는 장소는 {meet_where}.
B: 그녀의 외모는?
A: {sex_partner_str2}의 외모는 {appearance}.
B: 그녀의 재산은?
A: {sex_partner_str2}의 재산은 {money}.
B: 그녀의 이름은?
A: {sex_partner_str2}의 이름은 당연히 알 수가 없어.
"""
    fortune_prompt = f"""
B는 사주를 통해 연애상담을 하는 고객이다.
B는 {sex_str}인데 앞으로 만날 {sex_partner_str}에 대해서 궁금해서 온라인 채팅을 통해서 연애 상담중이다.
{sex_partner_str2}는 고객이 만날 수도 있는 미래의 연인이다.
{sex_partner_str2}의 정보는 아래와 같다.
{sex_partner_str2}는 싱글이다.
{sex_partner_str2}의 성별은 {sex_partner_str} 이다.
{sex_partner_str2}의 직업은 {job} 중 하나일 가능성이 높다.
{sex_partner_str2}의 성격은 {personality} 일 가능성이 있어.
{sex_partner_str2}을 만나는 시기는 {meet_when}.
{sex_partner_str2}을 만나는 장소는 {meet_where}.
{sex_partner_str2}의 외모는 {appearance}.
{sex_partner_str2}의 재산은 {money}.
{sex_partner_str2}의 이름은 당연히 알 수가 없어.
###
B: 누구세요?
A: 나는 명리학에 통달한 도사야. 너의 미래 파트너를 점지해 주는 중이지. 니가 미래에 만날 사람에 대해서 알려 주려는 거야.
B: {sex_partner_str2}는 어떤 사람이야?
A: {sex_partner_str2}의 성격은 {personality} 으로 보이네, 그리고 외모는 {appearance}. 또한 {sex_partner_str2}의 직업은 {job} 정도로 예상이 돼. 
B: {sex_partner_str2}에 대해서 말해줘.
A: {sex_partner_str2}의 재산은 {money} 그리고 성격은 {personality} 로 보여. 그리고 {sex_partner_str2}의 외모는 {appearance}
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

def article(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "article"  
    init_user_data(context)  
    update.message.reply_text("기사 작성 모드로 전환 되었습니다.")
    update.message.reply_text("기사 제목을 입력 하세요.")

def blog(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "blog"  
    init_user_data(context)  
    update.message.reply_text("블로그 작성 모드로 전환 되었습니다.")
    update.message.reply_text("블로그 제목을 입력 하세요.")

def receipe(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "receipe"  
    init_user_data(context)  
    update.message.reply_text("레시피 작성 모드로 전환 되었습니다.")
    update.message.reply_text("음식 이름을 입력 하세요.")

def poem(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "poem"  
    init_user_data(context)  
    update.message.reply_text("시 작성 모드로 전환 되었습니다.")
    update.message.reply_text("시 제목을 입력 하세요.")

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

def status(update, context: CallbackContext):
    status_sub(update.message, context)
    
def status_sub(message, context: CallbackContext):
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
        
block_list = ["61182118"]        
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
        
    if q == "/chatgpt":
        if 'chatgpt' in context.user_data:
            openai.api_base = basaran_api_base
            context.user_data.pop('chatgpt', None)
            message.reply_text("ChatGPT mode disabled.")
        else:
            openai.api_base = openai_api_base
            context.user_data["chatgpt"] = True
            message.reply_text("ChatGPT mode enabled.")
        return
    elif q == "/stop":
        context.user_data['stop_generation'] = True
        return
    elif q == "/newchat":
        context.user_data['stop_generation'] = True
        clear_chat_history(context)
        message.reply_text("새로운 대화를 시작합니다.")
        return
    elif q == "/regen":
        if 'last_user_input' in context.user_data:
            context.user_data['stop_generation'] = True
            clear_chat_history(context)
            q = context.user_data['last_user_input']
        else:
            message.reply_text("저장된 입력이 없습니다.")
            return
        
    elif q == "/saju":
        context.user_data["councelor_type"] = "saju"  
        init_user_data(context)  
        message.reply_text("사주 모드로 전환 되었습니다.")
        return
    elif q == "/adult":
        context.user_data["councelor_type"] = "dirty"  
        init_user_data(context)  
        message.reply_text("OPT-Erebus story mode.")
        return
            
    #print(f'{user_id}, {block_list}, {user_id in block_list}')
    if str(user_id) in block_list:
        print('blocked.')
        return
    
    if "councelor_type" not in context.user_data or "mode" not in context.user_data:
        context.user_data["councelor_type"] = "expert"
        context.user_data["language"] = "ko"
        init_user_data(context)
        # if username == 'ninedra9ons':
        #     context.user_data["mode"] = "testmode"
        #     #context.user_data["shownormal"] = True

        message.reply_text("언어모델이 재시작 되었습니다. 이전의 대화는 더이상 유효하지 않으며 새로운 대화가 시작 됩니다.")
        message.reply_text(f"/help 도움말\n/stop 생성 중지\n/newchat 새로운 대화 시작\n\nChangGPT-2023.3.25.21.40")
        # message.reply_text("저사양 GPU에서 동작중이라 응답속도가 느립니다. 긴 문장 생성에는 10초 이상이 걸릴 수도 있습니다.")
        # update.message.reply_text(HELP_TEXT)
        if username == 'ninedra9ons':
            status_sub(message, context)

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
    elif councelor_type == "article":
        content = f'{article_writing}제목: {q}\n기사:'
        prompt, generated = generate(context, message, content, True)
        return
    elif councelor_type == "blog":
        content = f'{blog_writing}제목: {q}\n블로그:'
        prompt, generated = generate(context, message, content, True)
        return
    elif councelor_type == "receipe":
        content = f'{receipe_writing}요리 이름: {q}\n만드는 법:'
        prompt, generated = generate(context, message, content, True)
        return
    elif councelor_type == "poem":
        content = f'{poem_writing}제목: {q}\n시:'
        prompt, generated = generate(context, message, content, True)
        return
    
    context.user_data['chat_id'] = chat_id
    
    q_start_time = datetime.today()
    q_start_time_str = q_start_time.strftime('%Y.%m.%d %H:%M:%S')
    prompt, a = query(context, message, q)
    context.user_data['last_user_input'] = q
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
        print('no generation or generation error******************')
        # clear_chat_history(context)
        # message.reply_text("잠깐만... 오류났다...")
        # prompt, a = query(context, message, q)
        # a = a.strip()
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
updater.dispatcher.add_handler(CommandHandler('article', article))
updater.dispatcher.add_handler(CommandHandler('blog', blog))
updater.dispatcher.add_handler(CommandHandler('receipe', receipe))
updater.dispatcher.add_handler(CommandHandler('poem', poem))

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

command = [
    BotCommand("newchat","생성을 중단하고 새로운 대화 시작"), 
    BotCommand("regen","답변 새로 생성"), 
    BotCommand("stop", "생성 중단")
]
bot = Bot(os.environ['TELEGRAM_LM_CHAT'])
if telegram_test_mode:
    bot = Bot(os.environ['TELEGRAM_LM_CHAT_TEST'])
bot.set_my_commands(command)

if not telegram and not facebook:
    print('error: no messenger server setted.')

if telegram:
    print("telegram bot started...")
    updater.start_polling()

if facebook:
    print("facebook bot started...")
    fb_messenger_start()    # should be last one