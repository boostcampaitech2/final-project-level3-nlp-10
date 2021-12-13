import time
import random
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

defaults = dict(learning_rate=0.0001, label_smoothing=0.1, epochs=50, embedding_dim=768, channel=512)
wandb.init(config=defaults, project='final_project')
config = wandb.config

def main():
    print(f'config : {defaults}')
    
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('emeraldgoose/bad-korean-tokenizer')

    df = pd.read_csv('data.csv')
    eval_df = pd.read_csv('korean_hate_speech.csv')
    
    labels = df['label']
    eval_labels = eval_df['label']
    df = punctuation(df)
    df = tokenized_sentence(tokenizer, df)
    eval_df = tokenized_sentence(tokenizer, eval_df)
    
    dataset = load_dataset(df, labels)
    eval_dataset = load_dataset(eval_df, eval_labels)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = modeling.Model(
        vocab_size=vocab_size, 
        max_seq=200, 
        embedding_dim=config.embedding_dim, 
        channel=config.channel, 
        num_class=2,
        device=device)
    model.to(device)
    wandb.watch(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    epochs = config.epochs
    best_f1 = 0.7
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        model.train()
        for i, batch in pbar:
            input = batch['input_ids'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input)

            loss = criterion(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

            epoch_loss = total_loss/(i+1)
            pbar.update()
            pbar.set_description(f"epoch {epoch+1}/{epochs}, loss : {epoch_loss:.3f}")
            wandb.log({'loss' : epoch_loss})
        pbar.close()

        # if (epoch+1)%10!=0: continue
        preds = []
        mean_time = 0
        with torch.no_grad():
            model.eval()
            for batch in eval_dataloader:
                start = time.time()
                
                input = batch['input_ids'].to(device)
                output = model(input)
                preds.append(output.argmax(-1).item())
                
                end = time.time()
                mean_time += (end - start)
        
        eval_f1 = f1_score(eval_labels, preds, average='micro')

        print(f"eval f1 : {eval_f1:.3f}, mean time : {mean_time/len(eval_dataset)}")

        if best_f1 < eval_f1 :
            torch.save(model.state_dict(), './save/result.pt')
            best_f1 = eval_f1

    print(f'best f1 = {best_f1}')

def evaluation(tokenizer, device):
    df = pd.read_csv('korean_hate_speech.csv')
    
    labels = df['label']
    df = tokenized_sentence(tokenizer, df)
    
    dataset = load_dataset(df, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = modeling.Model(
        vocab_size=vocab_size, 
        max_seq=200, 
        embedding_dim=config.embedding_dim,
        channel=config.channel,
        num_class=2,
        device=device)
    model.load_state_dict(torch.load('./save/result.pt'))
    model.to(device)

    preds = []
    mean_time = 0
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            start = time.time()
            
            input = batch['input_ids'].to(device)
            output = model(input)
            preds.append(output.argmax(-1).item())
            
            end = time.time()
            mean_time += (end - start)
    
    eval_f1 = f1_score(labels, preds, average='micro')
    eval_acc = accuracy_score(labels, preds)

    print(f"eval f1 : {eval_f1}, eval_acc : {eval_acc}, mean time : {mean_time/len(dataset)}")
    
    wandb.log({'f1' : eval_f1, 'mean_time' : mean_time/len(dataset)})


if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    main()

