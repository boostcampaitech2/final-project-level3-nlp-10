import torch
from torch.utils.data.dataloader import Dataset
    
    
class load_dataset(Dataset):
    """dataset class"""
    def __init__(self, dataset, labels) -> None:
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.dataset.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def punctuation(dataset):
    """punctuation preprocessing."""
    """텍스트 길이의 10%를 punctuation 삽입하여 모델이 robust하도록 한다."""
    punc = ['.',',','!','@','~','?','*','^','%']
    for i in range(len(dataset)):
        text = dataset['text'][i].split()

        if len(text) < 6: continue

        punc_size = max(round(len(text) * 0.1), 3) # 텍스트가 짧은 경우 3개만 들어가도록

        text_rnd = torch.randint(low=0, high=len(text)-1, size=(1,punc_size)).tolist()[0]
        punc_rnd = torch.randint(low=0, high=len(punc), size=(1,punc_size)).tolist()[0]
        
        for txt_rnd, pun_rnd in zip(text_rnd, punc_rnd):
            text.insert(txt_rnd, punc[pun_rnd])    
        
        dataset['text'][i] = ''.join(text)
    
    return dataset


def tokenized_sentence(tokenizer, dataset):
    """return tokenized sentence(input_ids, token_type_ids, attention_mask)"""
    tokenized = tokenizer(
        list(dataset['text']),
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=200,
        return_token_type_ids=False,
        add_special_tokens=True,
    )
    return tokenized