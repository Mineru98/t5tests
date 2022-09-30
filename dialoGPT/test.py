from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
import nltk
import re

nltk.download('punkt')

model_name = "dialoGPT-base-korean-chit-chat"
#model_checkpoint = 'byeongal/Ko-DialoGPT'
model_checkpoint = f"./Models/{model_name}/checkpoint-220000"   # restore and continue


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

print("running on", device)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_checkpoint)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint).to(device)

past_user_inputs = []
generated_responses = []

max_input = 512
# while True:
#     user_input = input(">> User: ")
#     if user_input == 'bye':
#         break
#     text_idx = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
#     for i in range(len(generated_responses)-1, len(generated_responses)-3, -1):
#         if i < 0:
#             break
#         encoded_vector = tokenizer.encode(generated_responses[i] + tokenizer.eos_token, return_tensors='pt')
#         if text_idx.shape[-1] + encoded_vector.shape[-1] < max_input:
#             text_idx = torch.cat([encoded_vector, text_idx], dim=-1)
#         else:
#             break
#         encoded_vector = tokenizer.encode(past_user_inputs[i] + tokenizer.eos_token, return_tensors='pt')
#         if text_idx.shape[-1] + encoded_vector.shape[-1] < max_input:
#             text_idx = torch.cat([encoded_vector, text_idx], dim=-1)
#         else:
#             break
#     text_idx = text_idx.to(device)
#     inference_output = model.generate(
#             text_idx,
#             max_length=max_input,
#             num_beams=3,
#             do_sample=True, 
#             # top_k=20,
#             # no_repeat_ngram_size=4,
#             # length_penalty=0.65,
#             # repetition_penalty=2.0,
#         )
#     inference_output = inference_output.tolist()
#     bot_response = tokenizer.decode(inference_output[0][text_idx.shape[-1]:], skip_special_tokens=True)
#     print(f"Bot: {bot_response}")
#     past_user_inputs.append(user_input)
#     generated_responses.append(bot_response)


chat_history = []
# Let's chat for 5 lines
for step in range(100):
    print("")
    user_input = input(">> User: ")
    chat_history.append(["User", user_input])
    while len(chat_history) > 7:
        chat_history.pop(0)
    # print(chat_history)
    hist = ""
    for chat in chat_history:
        hist += chat[1] + tokenizer.eos_token
    hist = hist[-max_input:]
    print("====", len(chat_history))
    print("===>", hist)
    print("----")
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(hist, return_tensors='pt').to(device)
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    #bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, max_length=bot_input_ids.shape[-1] + 100,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,       
        do_sample=True, 
        num_beams=5,
        early_stopping=True,
        repetition_penalty=2.0,
        length_penalty=0.65,
        top_k=20, 
        #top_p=0.7,
        #temperature = 0.8
    )

    bot_text = tokenizer.decode(chat_history_ids[0][bot_input_ids.shape[-1]:], skip_special_tokens=True).replace("#@이름#", "OOO")
    print("org=", bot_text)
    bot_text = re.sub("\\.\\.+", ". ", bot_text)
    print("remove ...=", bot_text)
    bot_text = nltk.sent_tokenize(bot_text)
    print("sentence=", bot_text)
    bot_text = bot_text[0]
    print("Bot: {}".format(bot_text))    
    chat_history.append(["Bot", bot_text])
    
    print("\nchat history")
    for chat in chat_history:
        print(f"{chat[0]}:\t{chat[1]}")
