# !pip install koco
import koco
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import re


def koco_info(train_dev, unlabeled, test, curse):
    print("="*20, "train_dev", "="*25)
    print()
    print("type of 'train_dev' dataset:", type(train_dev))
    print("keys in 'train_dev' dataset:", train_dev.keys())
    print("total length of 'train_dev' dataset:", len(train_dev['train'])+len(train_dev['dev']))
    print("sample data from 'train_dev' dataset")
    pprint(train_dev['train'][0])
    print()
    print("="*20, "unlabeled", "="*25)
    print()
    print("type of 'unlabeled' dataset:", type(unlabeled))
    print("total length of 'unlabeled' dataset:", len(unlabeled))
    print("sample data from 'unlabeled' dataset")
    pprint(unlabeled[0])
    print()
    print("="*22, "test", "="*25)
    print()
    print("type of 'test' dataset:", type(test))
    print("total length of 'test' dataset:", len(test))
    print("sample data from 'unlabeled' dataset")
    pprint(test[0])
    print()
    print("="*22, "curse", "="*25)
    print()
    print("type of 'curse' dataset:", type(curse))
    print("total length of 'curse' dataset:", len(curse))
    print("sample data from 'curse' dataset")
    pprint(curse[0])

# 복붙 도배 문장 걸러내는 함수
def principal_period(s):
    i = (s+s).find(s, 1, -1)
    return None if i == -1 else s[:i]

def check_repeat(user_chat):
    if principal_period(user_chat): # 복붙으로 도배한 문장 걸러서 통과 (띄어쓰기 없이 복붙)
        s = principal_period(user_chat)
        if len(s) <= 3: # 반복되는 문장이 3글자 이하면 1번 반복 시킴
            return s+s
        else: # 이외에는 반복 X
            return s
    
    elif principal_period(user_chat + " "): # 복붙으로 도배한 문장 걸러서 통과 (띄어쓰기 있는 복붙)
        s = principal_period(user_chat + " ")
        if len(s) <= 3:
            return s+s
        else:
            return s
        
    user_chat = user_chat[:-1] # 복붙 후 마지막에 v, ㅍ, ' 등이 붙는 경우
    
    # 위와 동일
    if principal_period(user_chat):
        s = principal_period(user_chat)
        if len(s) <= 3:
            return s+s
        else:
            return s
    
    elif principal_period(user_chat + " "):
        s = principal_period(user_chat + " ")
        if len(s) <= 3:
            return s+s
        else:
            return s
    
    return None # 해당 사항 없으면 None 리턴

def preprocess_beep(train_dev, test, unlabeled):

    limit_len = 24 # 글자 수 제한
    hate_list = []
    for txt in train_dev['train']:
        if len(txt['comments']) > limit_len:
            continue
        hate_list.append(txt['comments'])

    print("train_dev['train'] data finished")

    for txt in train_dev['dev']:
        if len(txt['comments']) > limit_len:
            continue
        hate_list.append(txt['comments'])
        
    print("train_dev['dev'] data finished")

    for txt in test:
        if len(txt['comments']) > limit_len:
            continue
        hate_list.append(txt['comments'])
        
    print("test data finished")

    for txt in tqdm(unlabeled):
        if len(txt['comments']) > limit_len:
            continue
        hate_list.append(txt['comments'])

    print("unlabeled data finished")

    preprocessed_list = []

    for t in tqdm(hate_list):

        user_chat = re.sub(r"\s+", r" ", t)
        user_chat = re.sub(r"[“”‘’]", r"\'", user_chat)
        user_chat = re.sub(r"[〈<＜「≪《『]", "<", user_chat)
        user_chat = re.sub(r"[〉>＞」≫》』]", ">", user_chat)
        user_chat = re.sub(r'[‥…]+', r'...', user_chat)
        user_chat = re.sub(r'[\?¿？]+', r'\?', user_chat)
        user_chat = re.sub(r'[!¡！]+', r'!', user_chat)
        user_chat = re.sub(r"([^\-—_=+,\./<>?\[\]{};:\'\"!@#$%\^&*\(\)₩`´~\|\\ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔゞァ-・ヽヾ゛゜ー一-龯\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\s]+)", r'', user_chat)
        
        if len(user_chat) == 0: continue
        
        elif user_chat[0] == '!': continue
        
        elif re.fullmatch(r"[ㄱㄲㅋㄷㄸㅎㅠㅜZEGzeg\-=\?!\d\s]+", user_chat): # z(ㅋ), e(ㄷ), ㅎ, G/g(gg) + 숫자만 있는 문장 제거
            continue
        
        elif re.fullmatch(r"[ぁ-ゔゞァ-・ヽヾ゛゜ー一-龯w\d\s]+", user_chat): # w, 숫자, 일본어만 있는 문장 제거
            continue
            
        elif re.fullmatch(r"[\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\d\s]+", user_chat): # 영어, 숫자, 중국어만 있는 문장 제거
            continue
                
        elif re.fullmatch(r"[₩/\\\(\)\[\]<>\-_=+@#$%\^&*,\.;:\'\"\?!\s]+", user_chat): # 문장부호, 특수기호만 있는 문장 제거
            continue
        
        rep = check_repeat(user_chat)
        if rep: user_chat = rep
        
        rep_num = 0
        while re.search(r"(.)\1{4,}", fr"{user_chat}"):  # 4번 이상 반복되는 글자 3번만 반복되도록 수정
            rep_num += 1
            if rep_num > 20:
                print(user_chat)
                break
            repeated = re.search(r"(.)\1{4,}", fr"{user_chat}").group()
            if repeated[0] == '^':
                user_chat = re.sub(r'\^+', r'^^^', user_chat)
            try:
                user_chat = re.sub(repeated, repeated[0]*3, user_chat)
            except: break
                
        user_chat = re.sub(r"\\", "", user_chat) # 백슬래쉬 제거
        
        preprocessed_list.append(user_chat)
        
    hate_list = list(set(preprocessed_list))
    df_hate = pd.DataFrame(hate_list, columns=['text'])
    beep_set = list(set(df_hate['text']))
    pd.DataFrame(beep_set, columns=['text']).to_csv("./beepData.tsv", index=False, sep='\t')


if __name__ == '__main__':
    train_dev = koco.load_dataset("korean-hate-speech", mode="train_dev")
    unlabeled = koco.load_dataset("korean-hate-speech", mode="unlabeled")
    test = koco.load_dataset("korean-hate-speech", mode="test")

    curse = []
    with open("./curse_detection.txt", "r") as f:
        for line in f.readlines():
            curse.append(line.strip())

    koco_info(train_dev, unlabeled, test, curse)

    preprocess_beep(train_dev, test, unlabeled)