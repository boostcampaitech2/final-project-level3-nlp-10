import torch
from torch.utils.data import Dataset
from tqdm import trange
import numpy as np
import re
    
    
class load_dataset(Dataset):
    """dataset class"""
    def __init__(self, dataset, labels=None) -> None:
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx) -> dict:
        if self.labels == None:
            return {'input_ids' : torch.tensor(self.dataset[idx])}
        
        item = {
            'input_ids': torch.tensor(self.dataset[idx]),
            'label': torch.tensor(self.labels[idx]),
        }
        return item

    def __len__(self) -> int:
        return len(self.dataset)


def punctuation(dataset):
    """punctuation preprocessing."""
    """텍스트 길이의 10%를 punctuation 삽입하여 모델이 robust하도록 한다."""
    print('start puncutation')
    punc = ['.',',','!','@','~','?','*','^','%']
    for i in trange(len(dataset)):
        text = dataset['text'][i].split()

        if len(text) < 6: continue

        punc_size = max(round(len(text) * 0.1), 3) # 텍스트가 짧은 경우 3개만 들어가도록

        text_rnd = torch.randint(low=0, high=len(text)-1, size=(1,punc_size)).tolist()[0]
        punc_rnd = torch.randint(low=0, high=len(punc), size=(1,punc_size)).tolist()[0]
        
        for txt_rnd, pun_rnd in zip(text_rnd, punc_rnd):
            text.insert(txt_rnd, punc[pun_rnd])    
        
        dataset['text'][i] = ''.join(text)
    
    return dataset


def tokenized_sentence(tokenizer, df):
    tokenized = []
    for text in list(df['text']):
        tokens = tokenizer.encode(text).ids
        if len(tokens) <= 200:
            for i in range(200-len(tokens)): tokens.append(0)
        elif len(tokens) > 200:
            for i in range(len(tokens)-200): tokens.pop()
        tokenized.append(tokens)
    return tokenized


quotes = re.compile(r"[“”‘’\"\']")
l_bracket = re.compile(r"[〈<＜「≪《『]")
r_bracket = re.compile(r"[〉>＞」≫》』]")
dots = re.compile(r'[‥…]+')
question = re.compile(r'[\?¿？]+')
exclamation = re.compile(r'[!¡！]+')
remainders = re.compile(r"([^\-—_=+,\./<>?\[\]{};:\'\"!@#$%\^&*\(\)₩`´~\|\\ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔゞァ-・ヽヾ゛゜ー一-龯\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\s]+)")
multiple_spaces = re.compile(r"\s+")

def preprocessing(dataset):
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
