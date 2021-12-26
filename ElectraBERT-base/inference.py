from tqdm import trange
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from tokenizers import BertWordPieceTokenizer

import argparse
import os


def exec(text: str, tokenizer) -> None:
    max_length = 256
    encoded = tokenizer.encode(text)
    encoded.pad(max_length)
    encoded.truncate(max_length)
    input_ids = torch.tensor([encoded.ids]).to(device)
    token_type_ids = torch.tensor([encoded.type_ids]).to(device)
    attention_mask = torch.tensor([encoded.attention_mask]).to(device)
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    softmax = torch.nn.Softmax(dim=1)(output['logits'])
    return (1 if output['logits'].argmax(-1)==1 else 0), softmax[0][0], softmax[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ElectraBert Inference')
    parser.add_argument('--data', type=str, help='data to evaluate with trained ElectraBert')
    args = parser.parse_args()
    
    tokenizer = BertWordPieceTokenizer(
        "./vocab.txt",
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##",
    )
    model = AutoModelForSequenceClassification.from_pretrained('./')
    model.eval()

    dev = pd.read_csv(f"./data/{args.data}.tsv", sep='\t')
    text = dev['text']

    text = text[text == text]
    text = text[text != None]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    labeling = []
    none = []
    curse = []
    for i in trange(len(text)):
        ret = exec(text[i], tokenizer)
        labeling.append(ret[0])
        none.append(round(float(ret[1]),3))
        curse.append(round(float(ret[2]),3))

    df = pd.DataFrame(columns=['text','none','curse'])
    df['text'] = text
    df['none'] = none
    df['curse'] = curse
    df.to_csv(f"./data/{args.data}WithElectra.tsv", sep='\t', index=False)