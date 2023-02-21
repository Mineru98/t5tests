import os
import sys
import json
from datetime import datetime

import requests
from flask import Flask, request
from waitress import serve

app = Flask(__name__)

# tunning facebook web hook to local
# ssh -R 8091:127.0.0.1:5000 lcw.plan4.house

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

    # endpoint for processing incoming messaging events

    data = request.get_json()
    # log(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text
                    is_echo = False
                    if "is_echo" in messaging_event["message"]:
                        is_echo = messaging_event["message"]["is_echo"]
                    if not is_echo:
                        send_message(sender_id, f"echo {message_text}")

                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):

    log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))
    params = {
        "access_token": fb_page_access_token
    }
    headers = {
        "Content-Type": "application/json"
    }
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
    
def start():
    port = 5000
    print(f'running on port {port}')    
    serve(app, host="0.0.0.0", port=port)
    
if __name__ == '__main__':
    #app.run(debug=True)
    start()