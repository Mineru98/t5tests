import pandas as pd
import requests

def get_todays_fortune(gender, birthday, target_date):
    url = f'https://fortune.stargio.co.kr:28080/todayLuck/woonse?gender={gender}&saju={birthday}&loveDate={target_date}'
    print(url)
    r = requests.get(url)
    tables = pd.read_html(r.text, encoding='utf-8') 
    t = tables[0] 
    #print(t)
    c = 0
    s = f"사주원국을 보자면 시주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 1
    s += f"일주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 2
    s += f"월주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 3
    s += f"년주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이다."       
    #print(s)
    saju = s 

    t = tables[1] 
    #print(t)
    c = 0
    s = f"들어온 날을 보자면 일주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 1
    s += f"월주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이고, "
    c = 2
    s += f"년주 천간은 {t[c][3]},{t[c][2]},{t[c][1]} 지지는 {t[c][4]},{t[c][5]},{t[c][6]} 이다."       
    #print(s)
    target_date_samju = s

    t = tables[2] 
    #print(t)
    s = t[0][1]
    #print(s)
    fortune = s
    
    return saju, target_date_samju, fortune

def simple_keyword(user_input):
    bytes = str.encode(user_input)
    keyword_len = len(today_fortune_keyword)
    s = 0
    for i in bytes:
        s += i
    today = datetime.today().day + datetime.today().month + datetime.today().year
    key1 = int((s * 12 * today) % keyword_len)
    key2 = int((s * 3.141592 * 7 * today) % keyword_len)
    key3 = int((s * 36.5 * today) % keyword_len)
    keystr = f'{today_fortune_keyword[key1]}, {today_fortune_keyword[key2]}'
