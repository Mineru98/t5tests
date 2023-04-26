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
    s = f"사주원국을 보자면 시주는 {t[c][3]}{t[c][4]}, "
    c = 1
    s += f"일주는 {t[c][3]}{t[c][4]}, "
    c = 2
    s += f"월주는 {t[c][3]}{t[c][4]}, "
    c = 3
    s += f"년주는 {t[c][3]}{t[c][4]}이다."       
    #print(s)
    saju = s 

    t = tables[1] 
    #print(t)
    c = 0
    s = f"오늘 일주는 {t[c][3]}{t[c][4]}, "
    c = 1
    s += f"월주는 {t[c][3]}{t[c][4]}, "
    c = 2
    s += f"년주는 {t[c][3]}{t[c][4]}이다."       
    #print(s)
    target_date_samju = s

    t = tables[2] 
    #print(t)
    s = t[0][1]
    #print(s)
    fortune = s
    
    return saju, target_date_samju, fortune

if __name__ == "__main__":
    get_todays_fortune('male', '199909090000', '20201010')