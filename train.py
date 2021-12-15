import os
import time
import random
import numpy as np
import pandas as pd

import torch
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import modeling
from data import load_dataset, punctuation, tokenized_sentence

import wandb

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
set_seed(42)

defaults = dict(
    dropout1=0.3,
    dropout2=0.4,
    learning_rate=0.0001,
    label_smoothing=0.5,
    epochs=50,
    embedding_dim=100,
    channel=32)
wandb.init(config=defaults, project='final_project')
config = wandb.config


def main(tokenizer):
    print(f'config : {defaults}')

    df = pd.read_csv('train_.csv')
    eval_df = pd.read_csv('test_.csv')
    
    labels = df['label']
    eval_labels = eval_df['label']
    df = punctuation(df)

    df = tokenized_sentence(tokenizer, df)
    eval_df = tokenized_sentence(tokenizer, eval_df)
    
    dataset = load_dataset(df, labels)
    eval_dataset = load_dataset(eval_df, eval_labels)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = modeling.Model(
        vocab_size=vocab_size, 
        embedding_dim=config.embedding_dim, 
        channel=config.channel, 
        num_class=2,
        dropout1=config.dropout1,
        dropout2=config.dropout2,
        device=device)
    model.to(device)
    wandb.watch(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs = config.epochs,
        max_lr=0.01,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,
    )

    epochs = config.epochs
    best_f1, threshold_best_f1 = 0, 0.7877
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        preds = []
        labels = []
        # if os.path.exists('./save/temp/result.pt'):
        #     model.load_state_dict(torch.load('./save/temp/result.pt'))
        model.train()
        for i, batch in pbar:
            input = batch['input_ids'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input)

            loss = criterion(output, label)
            preds += output.argmax(-1).tolist()
            labels += label.tolist()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()

            epoch_loss = total_loss/(i+1)
            
            pbar.update()
            pbar.set_description(f"epoch {epoch+1}/{epochs}, loss : {epoch_loss:.3f}")
            wandb.log({'loss' : epoch_loss})
        
        accuracy = accuracy_score(labels, preds)
        print(f'train acc : {accuracy:.3f}')
        pbar.close()

        preds = []
        mean_time = 0
        with torch.no_grad():
            model.eval()
            for batch in eval_dataloader:
                start = time.time()
                
                input = batch['input_ids'].to(device)
                output = model(input)
                preds += output.argmax(-1).tolist()
                
                end = time.time()
                mean_time += (end - start)
        
        eval_f1 = f1_score(eval_labels, preds, average='micro')

        print(f"eval f1 : {eval_f1:.3f}, mean time : {mean_time/len(eval_dataset):.4}")
        wandb.log({'f1' : eval_f1, 'mean_time' : mean_time/len(dataset)})

        if eval_f1 > best_f1:
            torch.save(model.state_dict(), f'./save/temp/result.pt')
            if eval_f1 > threshold_best_f1  :
                print('-------- best score!!! save file --------')
                torch.save(model.state_dict(), f'./save/{eval_f1:.4f}.pt')
            best_f1 = eval_f1

    print(f'best f1 = {best_f1}')
    wandb.log({'f1' : best_f1})


def curse(tokenizer):
    df = pd.read_csv('curse.csv')
    eval_df = pd.read_csv('test_.csv')

    labels = df['label']
    eval_labels = eval_df['label']
    df = punctuation(df)

    df = tokenized_sentence(tokenizer, df)
    eval_df = tokenized_sentence(tokenizer, eval_df)
    
    dataset = load_dataset(df, labels)
    eval_dataset = load_dataset(eval_df, eval_labels)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = modeling.Model(
        vocab_size=vocab_size, 
        embedding_dim=config.embedding_dim, 
        channel=config.channel, 
        num_class=2,
        dropout1=config.dropout1,
        dropout2=config.dropout2,
        device=device)
    model.load_state_dict(torch.load('./save/temp/result.pt'))
    model.to(device)
    wandb.watch(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs = config.epochs,
        max_lr=0.01,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,
    )

    epochs = config.epochs
    best_f1, threshold_best_f1 = 0, 0.7877
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        preds = []
        labels = []
        # if os.path.exists('./save/temp/result.pt'):
        #     model.load_state_dict(torch.load('./save/temp/result.pt'))
        model.train()
        for i, batch in pbar:
            input = batch['input_ids'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input)

            loss = criterion(output, label)
            preds += output.argmax(-1).tolist()
            labels += label.tolist()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()

            epoch_loss = total_loss/(i+1)
            
            pbar.update()
            pbar.set_description(f"epoch {epoch+1}/{epochs}, loss : {epoch_loss:.3f}")
            wandb.log({'loss' : epoch_loss})
        
        accuracy = accuracy_score(labels, preds)
        print(f'train acc : {accuracy:.3f}')
        pbar.close()

        preds = []
        mean_time = 0
        with torch.no_grad():
            model.eval()
            for batch in eval_dataloader:
                start = time.time()
                
                input = batch['input_ids'].to(device)
                output = model(input)
                preds += output.argmax(-1).tolist()
                
                end = time.time()
                mean_time += (end - start)
        
        eval_f1 = f1_score(eval_labels, preds, average='micro')

        print(f"eval f1 : {eval_f1:.3f}, mean time : {mean_time/len(eval_dataset):.4}")
        wandb.log({'f1' : eval_f1, 'mean_time' : mean_time/len(dataset)})

        if eval_f1 > best_f1:
            torch.save(model.state_dict(), f'./save/temp/result.pt')
            if eval_f1 > threshold_best_f1  :
                print('-------- best score!!! save file --------')
                torch.save(model.state_dict(), f'./save/{eval_f1:.4f}.pt')
            best_f1 = eval_f1

    print(f'best f1 = {best_f1}')
    wandb.log({'f1' : best_f1})

if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('jiho0304/bad-korean-tokenizer')

    main(tokenizer)
    curse(tokenizer)

