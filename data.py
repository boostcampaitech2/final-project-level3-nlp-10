import torch
from torch.utils.data import Dataset
from tqdm import trange
    
    
class load_dataset(Dataset):
    """dataset class"""
    def __init__(self, dataset, labels) -> None:
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx) -> dict:
        # item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        # item['label'] = torch.tensor(self.labels[idx])
        item = {
            'input_ids': torch.tensor(self.dataset[idx]),
            'label': torch.tensor(self.labels[idx]),
        }
        return item

    def __len__(self) -> int:
        return len(self.labels)


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