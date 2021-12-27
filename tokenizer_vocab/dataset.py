""" 
    Reference : https://gist.github.com/kse0202/9d3d8d519170064cefdd12fcb718afa0
"""
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from tokenizers import BertWordPieceTokenizer

quotes = re.compile(r"[“”‘’\"\']")
l_bracket = re.compile(r"[〈<＜「≪《『]")
r_bracket = re.compile(r"[〉>＞」≫》』]")
dots = re.compile(r'[‥…]+')
question = re.compile(r'[\?¿？]+')
exclamation = re.compile(r'[!¡！]+')
remainders = re.compile(r"([^\-—_=+,\./<>?\[\]{};:\'\"!@#$%\^&*\(\)₩`´~\|\\ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔゞァ-・ヽヾ゛゜ー一-龯\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\s]+)")
multiple_spaces = re.compile(r"\s+")

def preprocessing(dataset):
    
    for i in trange(len(dataset)):
        comment = dataset.iloc[i, 0]
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
        
        dataset.iloc[i, 0] = comment
        
    print("checking nan")
    print(sum(dataset['text'].isna()), "number of nan exist")
    dataset = dataset[dataset['text'].notna()]
    print("checking null")
    print(sum(dataset['text'].isnull()), "number of null exist")
    dataset = dataset[dataset['text'].notnull()]
    
    return dataset

def punctuation(dataset):
    
    """punctuation preprocessing."""
    """텍스트 길이의 10%~20%를 punctuation 삽입하여 모델이 robust하도록 한다."""
    new_dataset = []
    punc = ['.',',',"'",';','/','-','~','!','@','?','^',' ']
    
    for i in trange(len(dataset)):
        text = dataset[i]
        if len(text) <= 30 and random.random() < 0.5: # 0.5 확률로 띄어쓰기 없애기 (길이 30 이하 텍스트만)
            text = ''.join(text.split())
            
        if random.random() < 0.8: # 0.8 확률로 punctuation 추가
            punc_size = random.randint(max(len(text)//20, 3), max(len(text)//10, 3)) # 모든 텍스트에 최소 3개는 들어가도록
            text = list(text)
            for _ in range(punc_size):
                txt_rnd = random.randint(0, len(text)-1)
                punc_rnd = random.randint(0, len(punc)-1)
                text.insert(txt_rnd, punc[punc_rnd])
            new_dataset.append(''.join(text).replace("  ", " "))
        else:
            new_dataset.append(text)
    return new_dataset


if __name__ == '__main__':

    # tsv 파일들을 불러옵니다
    curse_dataset = pd.read_csv('curse.tsv', sep='\t')
    beep_dataset = pd.read_csv('beepData.tsv', sep='\t')
    twitch_dataset = pd.read_csv('chatData.tsv', sep='\t')

    # 불러온 tsv을 모아줍니다
    train_dataset = pd.concat([curse_dataset[['text']], beep_dataset, twitch_dataset[['text']]], ignore_index=True)
    train_dataset = train_dataset[train_dataset['text'].notna()]
    train_dataset = train_dataset[train_dataset['text'].notnull()]

    # 전처리를 한 번 거친 후, 각 문장들을 punctuation-robust하도록 100번 punctuation을 추가해줍니다
    train_df = preprocessing(train_dataset)
    text = train_df['text'].to_list()
    text_list = []
    for _ in range(100):
        text_list += punctuation(text)
    txt_file = '\n'.join(text_list)

    with open("./sentence.txt", "w", encoding="UTF-8") as f:
        f.write(txt_file)


    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    # special token에 unused token 300개를 추가해줍니다
    unused = [f'[unused{i}]' for i in range(300)]

    # load sentences
    tokenizer = BertWordPieceTokenizer(
        vocab=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##",
    )

    # character 단위로 많이 끊어지도록 설정해줍니다
    limit_alphabet = 10000
    vocab_size = 30000

    tokenizer.train(
        files='./sentence.txt', # 문장들을 모아놓은 파일 load
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens = special + unused,
        show_progress=True,
        limit_alphabet=limit_alphabet,
    )
    
    # save tokenizer.json, pretty=True로 두시면 json 형식이 보기좋게 저장됩니다
    tokenizer.save("./tokenizer.json", pretty=True)
    tokenizer.save_model('./')  # save vocab.txt