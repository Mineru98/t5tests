
client = Client(hf_tgi_api_base)
response = client.generate_stream(contents, max_new_tokens=1024, **kwargs)

# Stream Answer
temp_gen_text_concat = ""
temp_gen_text_concat_start_pos = 0
no_gen_count = 0
stopped = False
for event in response:
    gen_text = event.token.text
    # if len(gen_text) > 0:
    #     print(f"finish_reason = {event['choices'][0]['finish_reason']}, {gen_text}, {ord(gen_text[0])}")
    time.sleep(speed)
    if len(gen_text) == 0 and not stopped:
        no_gen_count += 1
        print(f"no gen text={no_gen_count}")
        if no_gen_count > 5:
            reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message, True)
            break
        continue
    no_gen_count = 0
    prev_len = len(gen_text_concat)
    gen_text_concat += gen_text
    if not stopped:
        gen_text_concat, stopped = search_stop_word(gen_text_concat)
    gen_text = gen_text_concat[prev_len:]
    if len(gen_text) > 0:
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
        stop_pos = len(gen_text_concat) - temp_gen_text_concat_start_pos + 1
        if stop_pos < 0:
            stop_pos = len(gen_text_concat)
        temp_gen_text_concat = temp_gen_text_concat[:stop_pos]
        gen_text_to_reply += temp_gen_text_concat
        reply_text(context, message, gen_text_to_reply, gen_text_concat, sent_message, True)
        break
