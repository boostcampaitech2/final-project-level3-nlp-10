import os
import gc
import time
import pandas as pd
from scipy.sparse.construct import random
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tokenizers import BertWordPieceTokenizer

import modeling
import modeling2
import modeling3
from utils import Config, set_seed, GOOGLE_APPLICATION_CREDENTIAL, MLFLOW_TRACKING_URI
from data import load_dataset, punctuation, tokenized_dataset

# import mlflow

# os.environ['GOOGLE_APPLICATION_CREDENTIALS']=GOOGLE_APPLICATION_CREDENTIAL
# os.environ['MLFLOW_TRACKING_URI']=MLFLOW_TRACKING_URI

set_seed(42)

config = Config(
    dropout1=0.3,
    dropout2=0.4,
    learning_rate=1e-3,
    label_smoothing=0.5,
    epochs=50,
    embedding_dim=100,
    channel=128)

def alpha_weight(step):
    """Pseudo Label에 대한 Loss 가중치"""
    T1 = 100
    T2 = 700
    af = 3.0
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af

def train(tokenizer, device) -> None:
    
    # Print Hyperparameters
    print(f'config : {config.__dict__}')
    # mlflow.log_params(config.__dict__)
    

    # Train Dataset
    df = pd.read_csv('labeled.csv')
    p_df = pd.read_csv('twitch.csv')
    eval_df = pd.read_csv('test2.csv')

    # pseudo labeling할 데이터 중 7만개를 샘플로 사용합니다.
    p_df = p_df.sample(frac=0.04, random_state=42).reset_index().drop(['index'], axis=1)
    print(f'len = {len(p_df)}')
    
    # Test Dataset은 labeled dataset의 20%의 비율로 가져온다.
    df = df.drop(eval_df.index).reset_index().drop(['index'], axis=1)

    df = punctuation(df)

    labels = list(df['label'])
    eval_labels = list(eval_df['label'])

    df = tokenized_dataset(tokenizer, df)
    p_df = tokenized_dataset(tokenizer, p_df)
    eval_df = tokenized_dataset(tokenizer, eval_df)
    
    
    dataset = load_dataset(df, labels)
    p_dataset = load_dataset(p_df)
    eval_dataset = load_dataset(eval_df, eval_labels)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    p_dataloader = DataLoader(p_dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    # Load model
    vocab_size = 30000
    print(f'vocab size = {vocab_size}')
    model = modeling.Model(
        vocab_size=vocab_size, 
        embedding_dim=config.embedding_dim, 
        channel=config.channel, 
        num_class=2,
        dropout1=config.dropout1,
        dropout2=config.dropout2)
    model.to(device)
    
    # set criterion, optimizer, schdeuler
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # first train 100 epoch
    print('-----First Training-----')
    epochs = 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs = epochs,
        max_lr=0.01,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,
    )

    # best_acc = 0
    # best_f1 = 0
    # for epoch in range(epochs):
    #     running_loss = 0
    #     model.cuda()
    #     model.train()
    #     for i, labeled in enumerate(dataloader):
    #         input = labeled['input_ids'].to(device)
    #         label = labeled['label'].to(device)

    #         output = model(input)
    #         loss = criterion(output, label)
    #         running_loss += loss.item()

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
        
    #     with torch.no_grad():
    #         # Evalutaion
    #         model.cpu()
    #         model.eval()
    #         correct = 0
    #         total_time = 0
    #         prediction = []
    #         for j, batch in enumerate(eval_dataloader):
    #             input = batch['input_ids'].cpu()
    #             label = batch['label'].cpu()
                
    #             start = time.time()
    #             output = model(input)
    #             preds = output.argmax(-1)
    #             prediction += preds.tolist()
    #             total_time += time.time() - start
    #             correct += (preds==label).sum().item()
            
    #         eval_acc = correct/len(eval_dataset)
    #         f1 = f1_score(eval_labels, prediction, average='macro')

    #     print(f'Epoch: {epoch+1} | Train Loss : {running_loss/len(dataloader):.5f} | Acc : {eval_acc:.5f} | F1 : {f1:.3f} | Mean Time(128 batch) : {total_time/len(eval_dataloader):.3f}')
    #     # mlflow.log_metric('train loss', running_loss/len(dataloader))
    #     # mlflow.log_metric('train acc', eval_acc)
        
    #     if best_f1 < f1:
    #         best_f1 = f1
    #         torch.save(model.state_dict(), './save/temp/result.pt')
    #         # best_model = model
    # print(f'best acc : {best_acc}')

    # mlflow.pytorch.log_model(best_model, 'model', registered_model_name="ToxicityText_not_pseudo")


    # Second train(pseudo labeling)
    print('------Second Training------')

    torch.cuda.empty_cache()
    gc.collect()

    step = 100
    epochs = 10
    threshold_acc = 0.90 # 적어도 이 이상 나와줘야 함
    model.load_state_dict(torch.load('./save/temp/result.pt'))
    for epoch in range(epochs):
        for i, unlabeled in enumerate(p_dataloader):
            input = unlabeled['input_ids'].to(device)
            
            # Pseudo Labeling
            model.eval()
            output_unlabeled = model(input)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            
            model.train()
            output = model(input)
            unlabeled_loss = F.cross_entropy(output, pseudo_labeled)

            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                # 50 step마다 Labeled dataset으로 학습
                for j, batch in enumerate(dataloader):
                    input = batch['input_ids'].to(device)
                    label = batch['label'].to(device)
                    output = model(input)
                    loss = F.cross_entropy(output, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                step += 1
        
        # epoch마다 Evalution 진행
        model.eval()
        correct, loss = 0, 0
        zero, one = 0, 0
        preds = []
        with torch.no_grad():
            # Evaluation
            for i, batch in enumerate(eval_dataloader):
                data = batch['input_ids'].cuda()
                labels = batch['label']
                output = model(data)
                predicted = torch.max(output,1)[1]
                
                zero += predicted.tolist().count(0)
                one += predicted.tolist().count(1)
                
                correct += (predicted==labels.cuda()).sum()
                loss += F.cross_entropy(output, labels.cuda()).item()
                preds += predicted.tolist()
        
        eval_f1 = f1_score(eval_labels, preds, average='macro')
        print(f'Epoch: {epoch+1} | Alpha : {alpha_weight(step)} | Train Loss : {loss/len(eval_dataloader):.5f} | Test Acc : {correct/len(eval_dataset):.5f} | Zero : {zero} | One : {one} | F1 : {eval_f1}')
        # mlflow.log_metric('eval loss', loss/len(eval_dataloader))
        # mlflow.log_metric('Acc', correct.item()/len(eval_dataset))

        eval_acc = correct/len(eval_dataloader)
        if eval_acc > threshold_acc:
            torch.save(model.state_dict(), f'./save/pseudo/result.pt')
            # mlflow.pytorch.log_model(model, 'model', registered_model_name="ToxicityText")
        
        model.train()

    

if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    # load tokenizer
    tokenizer = BertWordPieceTokenizer('./vocab_3.txt', lowercase=False)

    # train(tokenizer, device)
    train(tokenizer, device)
