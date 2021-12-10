import time
import pandas as pd

import torch
from torch.nn.functional import embedding
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange

import modeling
from data import load_dataset, punctuation, tokenized_sentence


def main(max_seq, embedding_dim, kernel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    
    tokenizer = BertTokenizer.from_pretrained('emeraldgoose/bad-korean-tokenizer')

    df = pd.read_csv('curse.csv', index_col=0)
    
    labels = df['label']
    df = punctuation(df)
    df = tokenized_sentence(tokenizer, df)
    
    dataset = load_dataset(df, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = modeling.Model(
        vocab_size=vocab_size, 
        max_seq=max_seq, 
        embedding_dim=embedding_dim, 
        kernel=kernel, 
        num_class=2)
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    
    for epoch in range(epochs):
        total_loss = 0
        for (i, batch) in tqdm(enumerate(dataloader)):
            input = batch['input_ids'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(input)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        epoch_loss = total_loss/len(dataloader.dataset)
        print(f"epoch {epoch+1}/{epochs}, loss : {epoch_loss}")

    torch.save(model.state_dict(), './save/result.pt')


def eval(max_seq, embedding_dim, kernel):

    device = torch.device('cpu')
    print(f'device = {device}')
    
    tokenizer = BertTokenizer.from_pretrained('emeraldgoose/bad-korean-tokenizer')

    df = pd.read_csv('validation.csv')
    
    labels = df['label']
    df = tokenized_sentence(tokenizer, df)
    
    dataset = load_dataset(df, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = modeling.Model(
        vocab_size=vocab_size, 
        max_seq=max_seq, 
        embedding_dim=embedding_dim, 
        kernel=kernel, 
        num_class=2)
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


if __name__ == "__main__":
    max_seq = 200
    embedding_dim = 256
    kernel = 256
    main(max_seq, embedding_dim, kernel)
    eval(max_seq, embedding_dim, kernel)