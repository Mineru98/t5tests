import os, glob
import transformers
import torch
#from gpt_j_8bit import GPTJBlock8, GPTJForCausalLM8, GPTJModel8, add_adapters
from transformers import AutoTokenizer, logging, pipeline, AutoModel, AutoModelForCausalLM
import argparse, evaluate
from datasets import load_dataset, load_from_disk 

pipe = False
compute_perplexity = False
max_output_length = 1024
min_output_length = 400

model_name = "GPT-j-6B-8bit-wikipedia-finetune"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help = "model name")
parser.add_argument("-l", "--local_model", help = "local model name")
parser.add_argument("-t", "--tokenizer", help = "tokenizer")
parser.add_argument("-p", "--path", help = "model path with tokenizer")
parser.add_argument("-c", "--chat_mode", help = "chatting mode")
args = parser.parse_args()
latest_model_dir = "none"
if args.local_model:
    print("=== param using local model", args.local_model)
    model_name = args.local_model
    model_dir = f"./Models/{model_name}"
    try:
        latest_model_dir = max(glob.glob(os.path.join(model_dir, 'checkpoint-*/')), key=os.path.getmtime)
        tokenizer_dir = latest_model_dir
    except:
        latest_model_dir = model_dir
if args.model:
    print("=== param model name", args.model)
    model_name = args.model
    latest_model_dir = model_name
    if model_name == "hivemind/gpt-j-6B-8bit":
        tokenizer_dir = "EleutherAI/gpt-j-6B"
    else:
        tokenizer_dir = latest_model_dir
if args.tokenizer:
    tokenizer_dir = args.tokenizer
    
if args.path:
    latest_model_dir = args.path
    tokenizer_dir = latest_model_dir
    
num_chat_history = 0
if args.chat_mode:
    num_chat_history = int(args.chat_mode) 

print("\n---------------------------")
print("model dir =\t", latest_model_dir)
print("tokenizer dir =\t", tokenizer_dir)
print("---------------------------\n")

logging.set_verbosity_error()

if compute_perplexity:
    print('start perplexity compute.')
    #data = load_dataset("lcw99/oscar-ko-only", split='train[:50]')
    #data.save_to_disk("./test_data")
    data = load_from_disk("./test_data")['text']

    #data = load_dataset("lcw99/oscar-ko-only")['train']['text'][:50]
    input_texts = [s[:1024] for s in data if s!='']

    perplexity = evaluate.load("perplexity", module_type="metric")
    result = perplexity.compute(model_id=latest_model_dir, predictions=input_texts, device="cuda")
    print(result)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
gpt = AutoModelForCausalLM.from_pretrained(
    latest_model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    # device_map='auto',
    # load_in_8bit=True,
).to(device, torch.float16)

text_generation = pipeline(
    "text-generation",
    model=gpt,
    tokenizer=tokenizer,
    device=0
)


chat_history = []
#chat_prompt = "A는 35세, 성별은 남자이고, 이름은 박길동, 대기업 다니는 직장인 입니다.\n아래 대화를 연결해 보시오.\n"
chat_prompt = """
아래 대화에 나오는 A의 인물 특징은 아래와 같습니다.
이름은 둘리. 녹색의 아기 케라토사우루스 그러니까 공룡이다. 외계인에게 납치되어 실험 대상이 된 대가로 초능력을 얻었다. 이후 1억여 년간 빙하에 갇혀 있다가 서울의 우이천으로 떠내려 왔다. 사람들이 빙하를 가져가 고길동의 집 근처 하천으로 떠밀려 왔다. 심성은 착하나 장난이 매우 심하다. 고길동의 집에 얹혀서 살지만 크게 말썽을 부리고 말대꾸를 한다. 친구로는 도우너, 또치등이 있다. 고길동의 아들로 희동이가 있는데 둘리와 친하다.  둘리가 희동이를 잘 돌보기 때문에 고길동의 아내인 박정자는 둘리를 옹호한다.
아래 대화를 연결해 보시오.
"""
while True:
    if num_chat_history == 0:
        print("\n")
        print("Input: ")
        contents = ""
        while True:
            try:
                line = input()
            except EOFError:
                break
            contents += f"{line}\n"    
        print("wait...")
    else:
        contents = chat_prompt
        if len(chat_history) == 0:
            contents += "A: 안녕하세요?\n"
            print(contents)
        else:
            for ch in chat_history:
                contents += f"B: {ch['user']}\nA: {ch['bot']}\n"
        user_input = input("B: ")
        user_input = user_input.strip()
        contents += f"B: {user_input}\nA: "
    contents = contents.strip()
    encoded_input = tokenizer(contents, return_tensors='pt').to(device)
    print(f"text={len(contents)}, token={encoded_input['input_ids'].size()}")
    input_length = encoded_input['input_ids'].size()[1]
    print(f'input_length={input_length}')
    if num_chat_history == 0:
        if input_length * 2 + 10 < max_output_length:
            max_length = input_length * 2 + 10
            if max_length < min_output_length:
                max_length = min_output_length
        else:
            max_length = max_output_length
    else:
        max_length = input_length + 100
    print(f'max_length={max_length}')
    if pipe:
        generated = text_generation(
            contents,
            max_length=max_length,
            do_sample=True,
            min_length=100,
            num_return_sequences=1,
            early_stopping=True,
            num_beams=3,
            # top_p=0.95,
            # top_k=50
        )
        print("\n")
        print(generated[0]['generated_text'])
    else:
        output_sequences = gpt.generate(
            encoded_input["input_ids"], 
            do_sample=True,
            num_beams=3,
            length_penalty=1.0,
            temperature=1.1,
            top_k=50,
            top_p=0.95,
            repetition_penalty=2.0,
            max_length=max_length
        )
        # print(output_sequences)
        output = output_sequences[0].tolist()
        try:
            stop = output.index(tokenizer.eos_token_id)
        except:
            stop = len(output)
        # print(output, stop)
        garbage = tokenizer.decode(output[stop:], skip_special_tokens=False)        
        print(garbage)        
        print("----")        
        prompt = tokenizer.decode(output[:input_length], skip_special_tokens=False)
        generated = tokenizer.decode(output[input_length:stop], skip_special_tokens=False).strip()
        if num_chat_history == 0:        
            print(prompt)
            print("----")        
            print(generated)
        else:
            stop_index = generated.find("B:")
            if stop_index < 0:
                bot_message = generated
            else:
                bot_message = generated[:stop_index].strip()
            chat_history.append({"user": user_input, "bot": bot_message})
            while len(chat_history) > num_chat_history:
                chat_history.pop(0)
            print(f"{prompt} {bot_message}")
            