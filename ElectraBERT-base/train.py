import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tokenizers import BertWordPieceTokenizer
import datasets
from datasets import load_dataset

# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange
import random
import argparse
import os

LR = 4e-5
EPOCHS = 10 # EPOCHS * 10 만큼 돌아감
BATCH_SIZE = 32
EVAL_STEP = 2450


def punctuation(dataset):
    
    new_dataset = []
    punc = ['.',',',"'",';','/','-','~','!','@','?','^',' ']
    
    for i in range(len(dataset)):
        text = dataset[i]
        
        # 띄어쓰기 단위로 글자 랜덤으로 배열
        t = text.split()
        n_t = ''
        for w in t:
            if random.random() < 0.5: # 0.8 확률로 랜덤 배열
                n_t = ' '.join([n_t, ''.join(random.sample(w, k=len(w)))])
            else:
                n_t = ' '.join([n_t, w])
        text = n_t.strip()
        
        # 0.5 확률로 띄어쓰기 없애기 (길이 30 이하 텍스트만)
        if len(text) <= 30 and random.random() < 0.5:
            text = ''.join(text.split())
            
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
        new_dataset.append(''.join(text).replace("  ", " "))
    return new_dataset


def tokenized_dataset(data, tokenizer):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    dataset = dict()
    max_length = 256
    for text in tqdm(data):
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

    
class CL_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.dataset.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    
def compute_metrics(preds):
    labels = preds.label_ids
    preds = preds.predictions.argmax(-1)
    return {'f1_score': f1_score(labels,preds), 'acc' : accuracy_score(labels,preds)}


class ImbalanceTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        curse_weight = sum(self.train_dataset.labels)/len(self.train_dataset)
        self.loss_weights = torch.Tensor([1, (1 - curse_weight) / curse_weight])
        self.loss_function = nn.CrossEntropyLoss(weight = self.loss_weights).to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.loss_function(logits, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        return (loss, outputs) if return_outputs else loss
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='ElectraBert Trainer')
    parser.add_argument('--data', type=str, help='data to train ElectraBert')
    parser.add_argument('--go-on', action='store_true', help='continue training ElectraBert')
    args = parser.parse_args()
    
    pretrained = "./" if args.go_on else "beomi/KcELECTRA-base"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    config = AutoConfig.from_pretrained("jiho0304/bad-korean-tokenizer")
    tokenizer = BertWordPieceTokenizer(
        "./vocab.txt",
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##",
    )
    
    train_dataset, val_dataset = load_dataset(
        'csv', 
        data_files={"train":[f'./data/{args.data}.tsv']}, 
        delimiter='\t',
        split=['train[:80%]', 'train[80%:]']
    )
    
    train_text = []
    train_label = []
    for _ in trange(10):
        train_text += punctuation(train_dataset['text'])
        train_label += [i for i in train_dataset['label']]
    # train_text += train_dataset['text']
    # train_label += train_dataset['label']
    print("tokenizing data")
    train_text = tokenized_dataset(train_text, tokenizer)
    
    val_text = val_dataset['text']
    val_label = val_dataset['label']
    val_text = tokenized_dataset(val_text, tokenizer)
    
    print("data tokenized")
    
    train_set = CL_Dataset(train_text, train_label)
    val_set = CL_Dataset(val_text, val_label)
    
    print("dataset prepared")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("./vocab.txt", "r") as f:
        vocab = f.read().splitlines()
    model.resize_token_embeddings(len(vocab))
    model.to(device)
    
    training_arguments = TrainingArguments(
        do_train=True,
        output_dir=f'./results/',
        save_total_limit=2,
        save_steps=EVAL_STEP,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.05,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE//4,
        logging_dir='./logs',
        logging_steps=EVAL_STEP,
        weight_decay=1e-6,
        evaluation_strategy='steps',
        eval_steps=EVAL_STEP,
        load_best_model_at_end=True,
        label_smoothing_factor=0.1,
    )
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    trainer = ImbalanceTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    model.save_pretrained('./')