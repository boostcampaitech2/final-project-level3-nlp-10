from math import trunc
from os import truncate
import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset
    
    
class load_dataset(Dataset):
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


def tokenized_sentence(tokenizer, dataset):
    tokenized = tokenizer(
        list(dataset['text']),
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=200,
    )
    return tokenized