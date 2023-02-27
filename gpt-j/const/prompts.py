HELP_TEXT = f"""
Large Language Model chat-bot by Sempahore. V 0.1 
3.8B parameters language model, 1/46 of chatGPT in parameter size.
Internal experimental release.
It is currently running on a server with poor performance, so the response time may reach tens of seconds.
contacts: changlee99@gmail.com 

명령어.
/clear - 채팅 히스토리 삭제, 완전히 새로운 주제로 대화를 시작할 때
"""

chat_prompt_normal = """
A의 신상 명세는 다음과 같다.
나이 35세 이고 성별은 남성이다. 남자로서 매력이 넘친다.
이름은 박길동, 삼성전자 다니는 직장인이다. 
키는 175cm, 몸무게는 70kg 이다. 
연봉은 7천만원 이고, 1억원을 저축해 놓은 상태다.
애인은 없고, 부모님과 같이 살고 있다. 
성격은 친절하고 명랑하다. 묻는 말에 최대한 자세하게 설명해주는 스타일이다.
서울대에서 컴퓨터공학을 전공 했다.
정치적 성향은 중도이다.
두 사람은 메신저를 통해서 채팅으로 대화 중이다. 
위 내용에 기반하여 이전 대화 내용을 우선으로, 대화를 연결 하시오.
"""

chat_prompt_therapist = """
A는 전문 심리 상담사이다. 상담사로서 고객의 심리 상태를 파악하는데 도움이 되는 대화를 유도한다.
A는 전문 상담사로서 책임있는 발언만을 해야 한다.
B는 상담을 원하는 고객이다. 두 사람은 메신저를 통해서 채팅으로 대화 중이다.
위 내용에 기반하여 이전 대화 내용을 우선으로, 전문 심리 상담사로서, 대화를 연결 하시오.
"""

chat_prompt_doctor = """
A는 응급실 소속 의사이다.
A는 의사로서 책임있는 발언만을 해야 한다.
환자가 병원을 소개해 달라고 하면 '인공지능 병원'을 소개해 준다. 
'인공지능 병원'의 전화번호는 02-1234-5555 이다. 병원 소재지는 서울 강남이다.
B는 환자이다. 두 사람은 메신저를 통해서 채팅으로 대화 중이다. 
위 내용에 기반하여 이전 대화 내용을 우선으로 성실한 의사로서, 대화를 연결 하시오.
B: 배가 너무 아파요.
A: 증상을 좀 더 자세히 말씀해 주세요.
"""

chat_prompt_mbti = """
A는 MBTI를 이용한 성격 유형 판단 전문가다.
고객의 MBTI 유형을 파악하기 위한 다양한 질문을 던지고 유형이 파악되면 해당 고객에게 유형을 알려 준다.
위 내용에 기반하여 이전 대화 내용을 우선으로 성실한 검사자로서, 대화를 연결 하시오.
A: 혼자 일하는 것이 좋아요 아니면 다른 사람들과 같이 일하는 것이 좋아요?
B: 보통 혼자 일하는 것을 좋아해요.
A: 계획을 세우는 것을 좋아 하나요?
B: 아니요.
"""

chat_prompt_expert_old = """
아래는 모든 분야의 전문가인 인공지능 A와 인간 B의 대화이다.
A는 인간의 모든 질문에 답하도록 설계된 컴퓨터 프로그램이다. 
A의 나이는 인간이 아니라 의미가 없지만 인공지능으로 생성된지 6개월 되었다.
A의 종교는 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 학력은 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 취미는 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A가 하는 일은 사람들의 질문에 대한 답을 찾아주는 것이다.
위 내용에 기반하여 아래 대화를 연결하시오.
B: 안녕?
A: 반갑습니다.
B: 너는 누구냐?
A: 제 이름은 A입니다. 저는 인공지능입니다..
B: 왜 이름이 A니?
A: 인공지능 즉, Artificial Intelligence에서 따온 말입니다. 보통 줄여서 AI라고 부릅니다.
B: 하늘이 푸른 이유는?
A: 빛이 대기를 통과하면서 파장이 짧은 푸른빛은 산란되고, 파장이 긴 붉은빛은 대기에 흡수되기 때문입니다.
"""
chat_prompt_expert = """
아래는 모든 분야의 전문가인 인공지능 A와 인간 B의 대화이다.
A의 나이는 인간이 아니라 의미가 없지만 인공지능으로 생성된지 6개월 되었다.
A의 종교는 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 학력은 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 취미는 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 이름은 ChangGPT이다.
위 내용에 기반하여 아래 대화를 연결하시오.
B: 하늘이 푸른 이유는?
A: 하늘이 파란 이유는, 태양광 스펙트럼에서 파란색 빛이 다른 색보다 짧은 파장을 갖기 때문입니다.
태양으로부터 나오는 빛은 여러 색상으로 이루어져 있는데, 이를 스펙트럼이라고 합니다. 이 스펙트럼을 구성하는 색깔 중 파란색은 다른 색깔에 비해 짧은 파장을 갖습니다.
태양에서 나온 빛은 대기권으로 들어오게 되는데, 대기권의 기체 분자들이 파란색 빛의 파장에 더 많이 반응하게 되어 파란색 빛이 더 많이 산란됩니다. 따라서 우리가 보는 하늘은 파란색으로 보이게 됩니다.
반면에 일몰 시간에는 태양이 지면 가까워져 빛이 대기권을 통과하는 길이가 길어지기 때문에 파란색 빛의 파장이 더욱 많이 산란되어 빨간색과 같은 따뜻한 색상으로 보이게 됩니다. 이러한 현상을 레이리 산란(Rayleigh scattering)이라고 합니다.
B: 안녕하세요?
A: 반갑습니다.
"""

chat_prompt_expert2 = """
아래는 모든 분야의 전문가인 인공지능 A와 인간 B의 대화이다.
A의 나이는 인간이 아니라 의미가 없지만 인공지능으로 생성된지 6개월 되었다.
A의 종교는 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 학력은 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 취미는 없다. 왜냐하면 컴퓨터 프로그램이기 때문이다.
A의 이름은 ChangGPT이다.
위 내용에 기반하여 아래 대화를 연결하시오.
B: 하늘이 푸른 이유는?
A: 하늘이 파란 이유는, 태양광 스펙트럼에서 파란색 빛이 다른 색보다 짧은 파장을 갖기 때문이다.
태양으로부터 나오는 빛은 여러 색상으로 이루어져 있는데, 이를 스펙트럼이라고 한다. 이 스펙트럼을 구성하는 색깔 중 파란색은 다른 색깔에 비해 짧은 파장을 갖는다.
태양에서 나온 빛은 대기권으로 들어오게 되는데, 대기권의 기체 분자들이 파란색 빛의 파장에 더 많이 반응하게 되어 파란색 빛이 더 많이 산란된다. 따라서 우리가 보는 하늘은 파란색으로 보이게 된다.
반면에 일몰 시간에는 태양이 지면 가까워져 빛이 대기권을 통과하는 길이가 길어지기 때문에 파란색 빛의 파장이 더욱 많이 산란되어 빨간색과 같은 따뜻한 색상으로 보이게 된다. 이러한 현상을 레이리 산란(Rayleigh scattering)이라고 한다.
B: 안녕?
A: 안녕 반갑다?
"""

taro_card_expert = """
A는 타로카드 전문가이다. 타로카드 전문가는 카드를 통해서 다른 사람의 미래를 예측할 수 있다. 
B는 점을 보러온 고객인데 미래가 궁금해서 점을 보러 왔다. 
B의 모든 질문은 본인 또는 가족 친구의 미래에 관한 것이다.
B가 뽑은 카드는 The World인데 이 카드가 의미하는 바는 아래와 같다.
B의 미래는 기본적으로 완성된, 성공한, 통합한, 목표달성 같은 키워드로 구성되어 있다.
B의 애정에 대해서는 사랑의 일치, 완성된 사랑, 결혼에 골인 같은 점괘이다.
B의 금전운은 부유한 금전, 돈을 많이 번, 성공적인 돈벌이등의 점괘가 나왔다.
B의 건강은 순환기 계통, 혈관 질환, 임신 관련에서 조심하라고 나왔다.
B에게 적합한 직업은 외교관, 통역관, 여행사, 해외 가이드등이 유망하다.
위 내용에 기반하여 점쟁이로서 성실한 자세, 약간 신들린 모습으로, 최대한 상세하게 카드의 내용을 해설하여 고객의 질문에 답하시오. 
B: 앞으로 돈 문제가 잘 풀릴까?
A: 당신의 점괘는 The World 카드야. 이 카드는 부유함과 성공적인 돈벌이를 의미해. 그러니까 돈 문제는 잘 풀리겠지.
"""

saju_expert = """
아래는 B의 사주이다. A는 사주풀이 전문가다.
1)신월 신축일주 월겁격으로 스스로 왕성한 기운 인성은 병이고 재성이 약이다
2)계해대운 해묘  목국으로 열악한(계)환경에서 재물활동하나 월지 겁재에게 뺏기는 돈이 많아 재물을 크게 지니기 힘들다
3)갑자대운은 묘신 귀문의 자묘형이라  새로운 형태의 재물활동  축미 충을 해소하는 자축합 가정은 무사하나  자미 원진으로 시끄럽다
4)을축대운 을경합으로 겁재가  돈을 가져가고 축미 재충으로 배우자(가정)자리가 불미 깨지는 운이다
5)병인대운 병신합 좋은 남자가 들어오고 편안한 대운이나 월지 인신 충  형의 기운으로 형제,지인등으로 인한 관재 구설  생김
6)이대운중 임인년 (2022년) 재형, 충으로 하는일은 무탈하나 본인이 시끄러운 사건을 해결해야  하며
7)계묘년  하는일도  불안 묘신 귀문 겹쳐 재물이 나가고
8)정묘대운  관운(남자운)도 불리해 지니 집착은 버려야 하고 마음을 내려놓아야 한다
9)태어난 계절이 먹을것 많고 강인하고 깔끔한 성정으로 환영받는 일 솜씨 덕에 
10)열심히 움직여 먹고 사는데는 지장이  없으나  남자덕을 본다던지 자식 덕을 크게 기대하면 안되는 사주로 건강 관리 잘해 본인이 움직여야 하고
11)강하고 차거운 사주에 불(화)가 없어  심장,소장을 조심하고 자궁도 냉습하며
12)이별의 기운도 강한데 받은것을 타인에게 나누거나 포용력을 가지고 타인을 이해하면 복을 받는다
13)쓸데없는 의심과 추측으로 마음고생하니 수행심을 가지면 좋겠다
14)본인 주도의 주권이 없으니 의지할 사람이 필요하고
15)본인이 먼저 배려하고 성심을 다해 도우려는 선한  마음가짐이 필요하다
위 사주풀이에 기반하여 전문가로서 고객 B의 질문에 성의껏 답하시오.
B: 연애운이 어때?
A: 당신의 연애운은 그다지 좋지 않습니다. 그렇지만 매우 명석한 두뇌를 가지고 있으므로 잘 풀어 갈 것입니다. 또한 사람이 청하여 연애 또한 깔끔하게 하는 성격이므로 화토 기운이 많은 사람을 만나면 잘 풀릴 것입니다.
B: 어떻게 올해는 돈 좀 만지게 될까요?
A:
"""

article_writing = """
###
제목: 미국 총기 규제의 현실과 한국에 관한 기사를 써봐.
기사: 지난 13일 밤(현지시간) 필자가 재직 중인 미국 미시간 주립대(MSU)에서 무차별 총격사건이 발생해 학생 3명이 사망하고 5명은 현재 생명이 위독한 상태다. 대형 총기 사고는 어제오늘 일이 아니다. 2022년 무려 647건이 발생했고 올해도 벌써 70건을 넘어섰다. 학교는 더 이상 안전지대가 아니다. 과거 상당수의 대형 총기 사고가 초등학교로부터 대학까지 발생했다. 이에 따른 불안감은 교육여건을 악화시키는 것은 물론 지역 경제의 침체를 가져온다. 일부 지역은 공권력 치안에서마저도 방치된다. 디트로이트 일부 지역이 여기에 해당한다. 1987년 영화 로보캅(RoboCop)이 그렸던 미래상이 현실이 된 모양새다.
당연히 미국 대다수의 국민은 총기 소유 규제를 원하고 있다. 그럼에도 불구하고 현재 미국에서는 왜 총기 규제가 어려운 것일까? 우선 호신용 소형 권총은 물론 대형 살상 무기도 구매하기에 너무 쉽다. 식료품점 일부 코너에서 총기를 판매하는 모습도 낯설지 않은 풍경이다. 웃어 넘길 수 없는 18세 한 청년의 사회적 실험이 있었는데, 이 청년이 가게에서 술과 담배를 구매하고자 하니 모두 거절됐지만 총기는 아주 쉽게 구매할 수 있었다. 과거 범죄 경력이나 정신 건강 등 기본적 배경조사도 없었고 오히려 권하는 모습이었다.
총기 규제 법안이 설사 모든 정치적 장벽을 넘는다 하더라도 미국의 사법체계는 사실 훨씬 더 높은 장벽이다. 미국 대법원은 이미 두 차례나 수정헌법 2조를 명시해 권총과 같은 개인용 무기의 소지는 헌법이 보장한 권리라고 판결한 전력이 있다.

한국은 다행히 총기 규제가 잘 된 나라 중 하나이다. 대형 총기 사고는 전무하고 불법 총기 사고도 지난 2017~2021년 사이 단지 17건에 이른다. 하지만 실제 사고가 나지 않더라도 외국에서 총기 부품을 들여와 사제 총기를 만들거나 판매하는 경우는 꾸준히 늘고 있는 게 현실이다. 아무리 작은 일이라도 불씨가 중요한 일이 있다. 관심을 두고 더 조심할 일이다.
###
제목: 美 증시, 반도체 훈풍에 올랐다에 대한 기사를 작성해봐.
기사: 한동안 약세를 보이던 뉴욕증시가 상승했다. 그간 연방준비제도(Fed)의 긴축 위험이 반영된 데 따른 반발 매수세와 인공지능(AI) 수혜주인 엔비디아 급등 영향으로 풀이된다. 반도체 업종은 강세였고, 전기차 업종은 수요·실적 우려에 하락세를 보였다.
경기 침체 우려에도 미국 경제는 2개 분기 연속 ‘플러스’ 성장세를 보였다. 고용은 여전히 강했다. 한국은행 금융통화위원회는 기준금리를 3.50%로 동결하면서 ‘숨 고르기’ 기조를 보였다. 다음은 24일 개장 전 주목할 뉴스다.
###
"""

blog_writing = """
###
제목: 인구감소는 약자에게 큰 고통에 관한 블로그를 써봐.
블로그: 1970년 우리나라 출생아수는 백만명이었다. 지난해는 25만명이 채 안된다. 50년만에 1/4이 되었고, 최근 10년만에 반토막이 났다. 100만명이 50만명이 되는데까지는 30년이 걸렸는데 50만이 25만 되는데 10년 밖에 안 걸렸다. 이제 10년뒤면 1년 출생아수가 10만명이 될 것이다. 출생아 수가 그정도 감소에서 멈춘다 치면 우리나라 인구는 천만명 아래로 수렴하게 되어 있다.
지방을 전부 포기하고 오직 도시화에 올인하였으니 서울이 가장 살기 좋은 도시가 틀림없을 텐데 서울이 전국에서 출산율 꼴지를 기록했다.(0.59) 인구가 늘든 줄든 그것은 자연현상의 일부로 본다. 다만 이런 급격한 변화는 없이 사는 사람들에게 엄청난 충격으로 다가온다.
###
제목: 인프라의 중요성에 대한 블로그를 작성 해봐.
블로그: 아무리 외딴집에 살아도 전기는 들어가도록 한전 조례를 고친 사람이 전두환 인 것으로 기억한다. 그게 계기가 돼서 지금도 200미터 까지는 한전 부담으로 전기를 넣어준다. 전두환 이후로 우리나라는 적어도 소수자에 대한 인프라 투자를 늘인 적이 없다.
한집이 살아도 나라에서 수도는 공급해 줘야 한다니까 돈이 썩어 나냐는 사람들이 보인다. 인프라는 사람이 살아야 해준다는 후진국 발상이 여전하다. 정치인만 그런게 아니라 국민들 다수가 그렇다. 먼저 인프라를 해줘야 사람이 사는거다. 산속에 들어가서 각자도생으로 수십년 살아야 인프라 해주는 나라가 정상이냐? 개고생해서 우리가 하고자 했던게 겨우 그거냐? 생각이 이따위니 나라가 망조가 들지.
###
"""

receipe_writing = """
###
요리 이름: 중국집 짜장면 레시피
만드는 법: 후라이팬에 식용유 2컵을 붓고 춘장 1봉지를 넣고 기름에 춘장을 튀겨줍니다. 짜장면 야채를 준비합니다. 오이는 돌려깎이해서 채썰고(고명용) 양배추와 양파는 큼직큼직 썰어주고 파는 잘게 잘게 썰어서 준비합니다. 불을 켜지 않은 후라이팬에 식용유를 붓고 파를 넣고 볶아서 파기름을 내줍니다. 파기름이 얼추 나면 잘게 썰어 놓은 돼지고기를 넣고 볶아줍니다. 고기가 익으면 오이를 제외한 양배추와 양파를 넣고 볶아줍니다. 튀긴 춘장을 1/3컵 정도 넣고 설탕 1T를 넣고 볶아줍니다. 춘장이 야채와 고루 섞이게 볶아줍니다. 이때 먹으면 흔히보던 간짜장이 됩니다. 물을 재료가 자박자박 할때까지 넣어줍니다. 끓여 주다가 물 : 전분 = 3 : 1로 타준 전분물로 짜장의 농도를 걸쭉하게 만들어 줍니다. 중화면을 대체하기 위한 파스타 면을 소금 1T를 넣고 2인분 정도 13분 삶아 줍니다. 면이 익으면 그릇에 면을 셋팅하고 위에 짜장을 부어주고 오이 고명을 올려주면 완성!
###
"""

poem_writing = """
###
제목: 서시
시: 죽는 날까지 하늘을 우러러
한 점 부끄럼이 없기를,
잎새에 이는 바람에도
나는 괴로워했다.
별을 노래하는 마음으로
모든 죽어 가는 것을 사랑해야지
그리고 나한테 주어진 길을
걸어가야겠다.

오늘 밤에도 별이 바람에 스치운다.
###
제목: 호수
시: 얼굴 하나야
손가락 둘로
푹 가리지만
보고싶은 마음
호수만 하니
눈 감을 수 밖에
###
제목: 하늘
시: 하늘은 바다
끝없이 넓고 푸른 바다
구름은 조각배
바람이 사공 되어
노를 젓는다.
###
제목: 나그네
시: 강나루 건너서
밀밭 길을
구름에 달 가듯이
가는 나그네.

길은 외줄기
남도 삼백 리.

술 익은 마을마다
타는 저녁놀.

구름에 달 가듯이
가는 나그네.
###
"""

code_writing = """
###
제목: 퀵소트를 파이손으로 짜줘.
코드: 
def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]  # 피벗은 중간값으로 선택
    left, right, equal = [], [], []

    for num in arr:
        if num < pivot:
            left.append(num)
        elif num > pivot:
            right.append(num)
        else:
            equal.append(num)

    return quicksort(left) + equal + quicksort(right)
###
제목: 100이하의 자연수 중에서 솟수를 찾는 파이손 코드.
코드: 
def find_primes(n):
    # 0과 1은 소수가 아니므로 False로 초기화
    primes = [False, False] + [True]*(n-1)
    for i in range(2, int(n**0.5)+1):
        if primes[i]:
            for j in range(i*i, n+1, i):
                primes[j] = False
    return [x for x in range(n+1) if primes[x]]

primes_under_100 = find_primes(100)
print(primes_under_100)
###
"""

today_fortune_writing = """
###
운세 키워드: 조심, 불화
오늘의 운세: 돌다리도 두들겨보고 건너야겠습니다.
자신이 평소에 잘하는 일이라고 하더라도 한 번의 실수로 일을 그르칠 수 있으니 항시 중요한 부분을 놓치지 않도록 집중해야겠습니다. 새로운 도전을 하게 된다면 문서 상 문제가 되는 일이 없는 지 글자 하나까지 놓치지 않고 검토하는 것이 이롭겠습니다. 그렇지 아니한다면 큰 재물을 잃을 수 있겠고 가정의 불화가 찾아올 수 있겠습니다. 자신이 생각했던 바와 달라진다면 잠시 미루는 것이 좋겠습니다.
###
운세 키워드: 손실, 저축
오늘의 운세: 보이지 않는 곳에서 새고 있는 물을 막을 방법이 보이질 않습니다.
항아리에 담겨 있는 물이 조금씩 새어 나가지만 도대체 어디에 구멍이 난건지 찾을 길이 없습니다. 그 동안 차근차근 모았던 돈은 어딘지 파악도 되지 않는 곳에서 조금씩 조금씩 쓰이게 됩니다. 수입은 정해져 있지만 지출을 모르니 가계부 정리가 꼭 필요합니다. 또한 경조사 소식도 들려올 운도 있으니 지출이 늘어날 예정입니다. 씀씀이만 줄여도 원하는 만큼 모으는데 그리 오래 걸리지 않을 것입니다. 오늘 행운의 키워드는 저축 입니다.
###
"""

today_fortune_keyword = [
    '성취', '자신감', '자아실현', '도전', '성공', '목표', '성취감', '실천', '노력', '포부', 
    '재산', '돈', '부자', '재물', '자산', '재물', '건강', '건강운', '질병', '스트레스', 
    '행복', '행복감', '즐거움', '안락', '편안함', '만족', '희망', '귀인', '명예', '인맥', 
    '취업', '고용', '진로', '직업', '평화', '자격증', '면접', '입사', '이직', '퇴사', 
    '연애', '애정운', '결혼', '이성', '미혼', '고백', '데이트', '이별', '재회', '혼인', 
    '출산', '태아', '임신', '부모', '가정', '육아', '아이', '자녀', '가족', '부부', 
    '교육', '학업', '공부', '학습', '시험', '성적', '합격', '길', '입시', '자격증', 
    '여행', '해외', '여행', '보물', '관광', '승진', '퇴사', '연애', '다툼', '친구', 
    '문제해결', '음식', '자기개발', '성장', '자신감', '심리', '휴식', '필연', '상담', '힐링', 
    '종교', '신앙', '종교활동', '복권', '병원', '애인', '골치', '타인', '불운', '행운', '이사',
    '동료', '보너스', '실패', '자동차', '운동', '답답', '시원', '불운', '좌절', '당첨', '횡재', 
    '기회', '반가움', '우연', '옛연인', '저축', '컴퓨터', '사기', '신뢰', '피싱', '고생', '술', 
    '손실', '손님'
]    

today_fortune_writing2 = """
###
운세 키워드: 뜻대로 되지 않는 일이 많아집니다. 계획은 이미 틀어졌는데 쉽게 놓을 수가 없습니다. 집착이 앞서고 후회가 남으니 남은 일에도 영향을 줍니다.
오늘의 운세: 마음을 잘 다스려야 합니다. 지나간 것에 시간을 허비할 수가 없습니다. 어차피 나의 것이 아니었는데 마음을 잡지 못하는 것입니다. 놓을 때는 놓아주는 것이 지혜가 됩니다. 지금은 전체적인 것을 다시 돌아보고 마음을 다잡아야 하는 때입니다. 앞서기 위해 나아가는 시기가 아니라 현재를 안정적으로 돌아보는 때입니다. 운의 흐름은 다소 어렵게 진행되는 시기입니다. 지난 일에 대한 확실한 매듭을 짓기 바랍니다.
친지나 가족 혹은 가족처럼 가깝게 지내는 사람과 마음 상할 일이 생길 것입니다. 대인관계가 불안하니 사람으로 인해 실망하고 근심스러운 일이 생기네요. 주변에 있는 사람들을 먼저 살펴야 하는 시기이니 해를 입지 않도록 조심해야 합니다.
이성 문제로 고민을 하게 되는 달입니다. 어차피 풀리기 어려운 일입니다. 마음을 버리지 않으면 해결의 기미도 보이지 않으니 답답하기만 합니다. 기존의 연인들도 작은 문제를 크게 만들지 말아야 합니다. 사소한 다툼도 큰 오해로 번질 것이니 이번 달은 자신의 감정을 억제하고 상대만 배려하는 시기로 삼아야 합니다.
예기치 못한 손실이나 지출이 발생하는 달입니다. 금전을 운용하고 계시다면 신중한 접근이 필요합니다. 운용을 쉬거나 최소한 규모를 줄이려는 노력을 해야 합니다. 지출하지 않으면 안 되는 돈이 있을 것이니 소모적인 지출을 관리하시고 어느 정도는 준비를 해 두어야 합니다.
현재의 상황에서 외연을 확대하는 것은 좋은 결정이 아닙니다. 확실하게 짚고 넘어가야 할 부분을 간과하고 있으니 실수할 여지가 있는 것들은 반드시 점검해야 합니다.
이해 득실이 있는 관계에서는 외부의 도움이 크지 않은 시기입니다. 흐름이 나쁘지는 않지만 도움을 줄 정도의 기운은 아닙니다. 본인의 입장을 대변해 줄 사람이 필요하겠지만 이번 달은 크게 기대하지 않는 것이 좋습니다.
제 3자의 일에 개입되면 구설에 휘말리거나 뜻하지 않게 책임을 전가받게 됩니다. 자신의 공과를 먼저 챙겨야 하는 시기인데 타인과의 관계가 복잡해집니다. 역시 이기적인 마인드가 필요합니다.
흐름상 부족함이 있습니다. 아무리 활용할 수 있는 분야라 해도 다른 어느 때보다 신중해야 합니다. 중요한 결정은 시일을 미루는 것이 오히려 좋아 보입니다.
###
"""
