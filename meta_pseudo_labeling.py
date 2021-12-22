import os
import gc
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from transformers import ElectraForSequenceClassification
from tokenizers import BertWordPieceTokenizer

import modeling
from utils import Config, set_seed, GOOGLE_APPLICATION_CREDENTIAL, MLFLOW_TRACKING_URI
from data import load_dataset, punctuation, tokenized_dataset
from tqdm import tqdm

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=GOOGLE_APPLICATION_CREDENTIAL
os.environ['MLFLOW_TRACKING_URI']=MLFLOW_TRACKING_URI

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

    # pseudo labeling할 데이터 중 7만개를 샘플로 사용합니다.
    p_df = p_df.sample(frac=0.1, random_state=42).reset_index().drop(['index'], axis=1)
    
    # Test Dataset은 labeled dataset의 20%의 비율로 가져온다.(false/true ratio = 0.76)
    eval_df = df.sample(frac=0.2, random_state=42).reset_index().drop(['index'], axis=1)
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

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    p_dataloader = DataLoader(p_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # Load model
    vocab_size = 30000
    print(f'vocab size = {vocab_size}')
    student = modeling.Model(
        vocab_size=vocab_size, 
        embedding_dim=config.embedding_dim, 
        channel=config.channel, 
        num_class=2,
        dropout1=config.dropout1,
        dropout2=config.dropout2)
    student.to(device)

    teacher = ElectraForSequenceClassification.from_pretrained('jiho0304/curseELECTRA')
    teacher.to(device)
    
    # Set teacher, student's optimizer
    optimizer_s = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    optimizer_t = torch.optim.AdamW(teacher.parameters(), lr=config.learning_rate)

    # Meta Pseudo Labeling
    print('------Start Training------')

    torch.cuda.empty_cache()
    gc.collect()

    epochs = 99
    best_acc = 0
    for epoch in range(epochs):
        for i, (unlabeled, labeled) in tqdm(enumerate(zip(p_dataloader, dataloader))):
            u_input = unlabeled['input_ids'].to(device)
            attention_mask = unlabeled['attention_mask'].to(device)
            
            # teacher는 Pseudo Label을 생성하고 student가 학습
            teacher.eval()
            output_unlabeled = teacher(input_ids=u_input, attention_mask=attention_mask)
            _, pseudo_labeled = torch.max(output_unlabeled['logits'], 1)
            
            student.train()
            u_output = student(u_input)
            unlabeled_loss = F.cross_entropy(u_output, pseudo_labeled)

            optimizer_s.zero_grad()
            unlabeled_loss.backward()
            optimizer_s.step()

            # student는 Labeled dataset으로 학습하고 teacher 업데이트
            l_input = labeled['input_ids'].to(device)
            label = labeled['label'].to(device)
            l_output = student(l_input)
            labeled_loss = F.cross_entropy(l_output, label)

            optimizer_t.zero_grad()
            labeled_loss.backward()
            optimizer_t.step()
            
            tqdm.update()
            tqdm.set_description(f"epoch {epoch+1}/{epochs}")
        
            # epoch마다 Evalution 진행
            student.eval()
            correct, loss = 0, 0
            zero, one = 0, 0
            with torch.no_grad():
                for _, batch in tqdm(enumerate(eval_dataloader)):
                    data = batch['input_ids'].cuda()
                    labels = batch['label']
                    output = student(data)
                    predicted = torch.max(output,1)[1]
                    
                    zero += predicted.tolist().count(0)
                    one += predicted.tolist().count(1)
                    
                    correct += (predicted==labels.cuda()).sum()
                    loss += F.cross_entropy(output, labels.cuda()).item()

                    tqdm.update()
                    tqdm.set_description(f'Epoch: {epoch+1} | Train Loss : {loss/len(eval_dataloader):.5f} | Test Acc : {correct/len(eval_dataset):.5f} | Zero : {zero} | One : {one}')
            
            # print(f'Epoch: {epoch+1} | Train Loss : {loss/len(eval_dataloader):.5f} | Test Acc : {correct/len(eval_dataset):.5f} | Zero : {zero} | One : {one}')
            # mlflow.log_metric('eval loss', loss/len(eval_dataloader))
            # mlflow.log_metric('Acc', correct.item()/len(eval_dataset))

            eval_acc = correct/len(eval_dataloader)
            if eval_acc > best_acc:
                torch.save(student.state_dict(), f'./save/pseudo/result.pt')
                # mlflow.pytorch.log_model(student, 'model', registered_model_name="ToxicityText")
                best_acc = eval_acc
            
            student.train()

    

if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    # load tokenizer
    tokenizer = BertWordPieceTokenizer('./vocab_3.txt', lowercase=False)

    # train(tokenizer, device)
    train(tokenizer, device)
