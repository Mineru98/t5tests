from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re, os
import nltk

import logging

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

model_dir = "lcw99/t5-base-korean-chit-chat"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

history = []
def dialoGPT_korean(update, context):
    global history
    context.bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    user_input = update.message.text

    if user_input == "/":
        history = []
        return
    # tokenize the new input sentence
    hist = ""
    for chat in history[-1:]:
        hist += "A: " + chat[0] + tokenizer.eos_token + "B: " + chat[1] + tokenizer.eos_token
    hist += "A: " + user_input + tokenizer.eos_token + "B: "
    print("input=", hist)
    new_user_input_ids = tokenizer.encode(hist, return_tensors='pt')
    new_user_input_ids = new_user_input_ids[:512]

    # append the new user input tokens to the chat history
    bot_input_ids = new_user_input_ids

    # generate a response 
    chat_history_ids = model.generate(
        bot_input_ids, max_length=200,
        pad_token_id=tokenizer.eos_token_id,  
        #no_repeat_ngram_size=3,       
        do_sample=True, 
        #top_k=100, 
        #top_p=0.7,
        #temperature = 0.1
    )

    # convert the tokens to text, and then split the responses into lines
    bot_text = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True).replace("#@이름#", "OOO")
    bot_text = bot_text.replace("\n", " / ")
    print("sentence=", bot_text)

    history.append((user_input, bot_text))

    update.message.reply_text(bot_text)

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(os.environ.get('DIALOGPT_TELE_BOT_TOKEN'), use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, dialoGPT_korean))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()