"""
    데이터 전처리와 데이터셋을 구성하는 코드입니다.
"""

import re
import random
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import trange
    
    
class load_dataset(Dataset):
    """dataset class"""
    def __init__(self, dataset, labels=None) -> None:
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx) -> Dict:
        if self.labels == None:
            item = dict(
                input_ids=self.dataset['input_ids'][idx], 
                attention_mask=self.dataset['attention_mask'][idx])
        else:
            item = dict(
                input_ids=self.dataset['input_ids'][idx],
                attention_mask=self.dataset['attention_mask'][idx], 
                label=torch.tensor(self.labels[idx]))
        return item

    def __len__(self) -> int:
        return len(self.dataset['input_ids'])


def punctuation(dataset) -> pd.DataFrame:
    """punctuation를 단어 사이에 삽입. Weak augmentation"""""
    punc = ['.',',','!','@','~','?','*','^','%']
    for i in trange(len(dataset)):
        text = dataset['text'][i].split()

        if len(text) < 6: continue # 텍스트 길이가 너무 짧은 경우 건너뛴다

        punc_size = max(round(len(text) * 0.1), 3) # 텍스트가 짧은 경우 3개만 들어가도록 한다

        text_rnd = torch.randint(low=0, high=len(text)-1, size=(1,punc_size)).tolist()[0]
        punc_rnd = torch.randint(low=0, high=len(punc), size=(1,punc_size)).tolist()[0]
        
        for txt_rnd, pun_rnd in zip(text_rnd, punc_rnd):
            text.insert(txt_rnd, punc[pun_rnd])    
        
        dataset['text'][i] = ''.join(text)
    
    return dataset


def punctuation2(dataset) -> pd.DataFrame:
    """punctuation을 글자 사이에 삽입 + 단어 순서를 뒤바꾸거나 띄어쓰기 제거. Strong augmentation"""
    new_dataset = pd.DataFrame(columns=['text'])
    punc = ['.',',',"'",';','/','-','~','!','@','?','^',' ']
    for i in trange(len(dataset)):
        text = dataset[i]
        # 띄어쓰기 단위로 글자 랜덤으로 배열
        t = text.split()
        n_t = ''
        for w in t:
            if random.random() < 0.8: # 0.8 확률로 랜덤 배열
                n_t = ' '.join([n_t, ''.join(random.sample(w, k=len(w)))])
            else:
                n_t = ' '.join([n_t, w])
        text = n_t.strip()
        
        # 0.5 확률로 띄어쓰기 없애기 (길이 30 이하 텍스트만)
        if 0 < len(text) <= 30 and random.random() < 0.5:
            text = ''.join(text.split())
        
        if len(text)==0: text+='ㅋ'
        # 랜덤으로 앞이나 뒤에 'ㅋㅋㅋ', 'ㅎㅎㅎ' 추가
        if text[0] != 'ㅋ' and text[0] != 'ㅎ' and random.random() < 0.5:
            add_front = random.choice(['ㅋㅋ', 'ㅋㅋㅋ', 'ㅎㅎ', 'ㅎㅎㅎ'])
            text = ''.join([add_front, random.choice([' ', '']), text])
            
        if text[-1] != 'ㅋ' and text[-1] != 'ㅎ' and random.random() < 0.5:
            add_last = random.choice(['ㅋㅋ', 'ㅋㅋㅋ', 'ㅎㅎ', 'ㅎㅎㅎ'])
            text = ''.join([text, random.choice([' ', '']), add_last])
            
        # punctuation 추가
        punc_size = random.randint(max(len(text)//10, 3), max(len(text)//5, 3))
        text = list(text)
        for _ in range(punc_size):
            txt_rnd = random.randint(0, len(text)-1)
            punc_rnd = random.randint(0, len(punc)-1)
            text.insert(txt_rnd, punc[punc_rnd])
        new_dataset = new_dataset.append(
            dict(text=''.join(text).replace("  ", " ")), ignore_index=True)
    
    return new_dataset


def tokenized_dataset(tokenizer, data) -> dict:
    """BertWordPieceTokenizer의 사용으로 dict(input_ids, token_type_ids, attention_mask) 리턴"""
    input_ids = []
    token_type_ids = []
    attention_mask = []
    dataset = dict()
    max_length = 200
    for text in data['text']:
        encoded = tokenizer.encode(text)
        encoded.pad(max_length)
        encoded.truncate(max_length)
        input_ids.append(encoded.ids)
        token_type_ids.append(encoded.type_ids)
        attention_mask.append(encoded.attention_mask)
        
    dataset['input_ids'] = torch.tensor(input_ids)
    dataset['token_type_ids'] = torch.tensor(token_type_ids)
    dataset['attention_mask'] = torch.tensor(attention_mask)
    return dataset


# 전처리를 위한 정규식 컴파일
quotes = re.compile(r"[“”‘’\"\']")
l_bracket = re.compile(r"[〈<＜「≪《『]")
r_bracket = re.compile(r"[〉>＞」≫》』]")
dots = re.compile(r'[‥…]+')
question = re.compile(r'[\?¿？]+')
exclamation = re.compile(r'[!¡！]+')
remainders = re.compile(r"([^\-—_=+,\./<>?\[\]{};:\'\"!@#$%\^&*\(\)₩`´~\|\\ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔゞァ-・ヽヾ゛゜ー一-龯\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\s]+)")
multiple_spaces = re.compile(r"\s+")


def preprocessing(dataset) -> pd.DataFrame:
    """
    dataset은 pd.DataFrame 형식으로 첫번째 column이 'text'인 경우를 상정하고 만들었습니다.
    """
    for i in trange(len(dataset)):
        comment = dataset[i]
        # checking nan
        if comment != comment:
            continue
        
        tmp = comment
        comment = quotes.sub(r"\'", comment)
        comment = l_bracket.sub("<", comment)
        comment = r_bracket.sub(">", comment)
        comment = dots.sub(r'...', comment)
        comment = question.sub(r'\?', comment)
        comment = exclamation.sub(r'!', comment)
        comment = remainders.sub(r'', comment)
        comment = multiple_spaces.sub(r" ", comment)
        if len(comment) <= 1 and tmp != comment:
            comment = np.nan
        elif len(comment) > 500:
            comment = np.nan
        
        dataset[i] = comment
    
    return dataset
