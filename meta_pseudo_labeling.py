import os
import gc
import pandas as pd
import numpy as np
from scipy.sparse import construct
from sklearn.metrics import f1_score

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from transformers import ElectraForSequenceClassification
from tokenizers import BertWordPieceTokenizer
from transformers.utils.dummy_pt_objects import get_polynomial_decay_schedule_with_warmup

import modeling
from utils import Config, set_seed, GOOGLE_APPLICATION_CREDENTIAL, MLFLOW_TRACKING_URI
from data import load_dataset, punctuation, tokenized_dataset
from tqdm import trange, tqdm

# os.environ['GOOGLE_APPLICATION_CREDENTIALS']=GOOGLE_APPLICATION_CREDENTIAL
# os.environ['MLFLOW_TRACKING_URI']=MLFLOW_TRACKING_URI

set_seed(42)

config = Config(
    dropout1=0,
    dropout2=0,
    learning_rate=1e-3,
    label_smoothing=0,
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

    # pseudo labeling할 데이터 중 10만개를 샘플로 사용합니다.
    true_label = p_df[(p_df['none']<p_df['curse'])==True]
    false_label = p_df.drop(true_label.index, axis=0).reset_index().drop(['index'], axis=1)
    false_label = false_label.sample(frac=0.1, random_state=42)
    true_label = true_label.reset_index().drop(['index'], axis=1)
    true_label = true_label.append(false_label)
    true_label = true_label.sample(frac=1, random_state=42).reset_index().drop(['index'], axis=1)
    p_df = true_label

    # Augmentation
    a_df = p_df
    
    # Test Dataset은 labeled dataset의 20%의 비율로 가져온다.(false/true ratio = 0.76)
    eval_df = df.sample(frac=0.2, random_state=42).reset_index().drop(['index'], axis=1)
    df = df.drop(eval_df.index).reset_index().drop(['index'], axis=1)

    df = punctuation(df)
    a_df = punctuation(a_df)

    labels = list(df['label'])
    eval_labels = list(eval_df['label'])

    df = tokenized_dataset(tokenizer, df)
    p_df = tokenized_dataset(tokenizer, p_df)
    a_df = tokenized_dataset(tokenizer, a_df)
    eval_df = tokenized_dataset(tokenizer, eval_df)
    
    dataset = load_dataset(df, labels)
    p_dataset = load_dataset(p_df)
    a_dataset = load_dataset(a_df)
    eval_dataset = load_dataset(eval_df, eval_labels)

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # p_dataloader = DataLoader(p_dataset, batch_size=1, shuffle=True)
    # a_dataloader = DataLoader(a_dataset, batch_size=1, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # Load model
    teacher = ElectraForSequenceClassification.from_pretrained('jiho0304/curseELECTRA')

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
    teacher.to(device)
    
    # Set teacher, student's optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_s = torch.optim.AdamW(student.parameters(), lr=1e-5)
    optimizer_t = torch.optim.AdamW(teacher.parameters(), lr=1e-5)

    # Meta Pseudo Labeling
    print('------Start Training------')

    torch.cuda.empty_cache()
    gc.collect()

    best_acc = 0
    dataset_iter, p_dataset_iter = 0, 0
    for step in trange(len(p_dataset)):
        teacher.train()
        student.train()

        try:
            labeled = dataset[dataset_iter]
            dataset_iter += 1
        except:
            dataset_iter = 0
            labeled = dataset[dataset_iter]
            dataset_iter += 1
        try:
            a_labeled = a_dataset[p_dataset_iter]
            unlabeled = p_dataset[p_dataset_iter]
            p_dataset_iter += 1
        except:
            p_dataset_iter = 0
            a_labeled = a_dataset[p_dataset_iter]
            unlabeled = p_dataset[p_dataset_iter]
            p_dataset_iter += 1

        l_input = labeled['input_ids'].tolist()
        l_attention_mask = labeled['attention_mask'].tolist()
        targets = labeled['label'].to(device)

        # strong augmentation을 augmentation된 unlabeled dataset으로 가정
        a_input = a_labeled['input_ids'].tolist()
        a_attention_mask = a_labeled['attention_mask'].tolist()

        # weak augmentation을 original unlabeled dataset으로 가정
        u_input = unlabeled['input_ids'].tolist()
        u_attention_mask = unlabeled['attention_mask'].tolist()

        t_input_ids = torch.tensor([l_input, a_input, u_input]).to(device)
        t_attention_mask = torch.tensor([l_attention_mask, a_attention_mask, u_attention_mask]).to(device)
        
        t_logits = teacher(input_ids=t_input_ids, attention_mask=t_attention_mask)['logits']
        t_logits_l, t_logits_a, t_logits_u = t_logits
        
        t_loss_l = criterion(t_logits_l.unsqueeze(0), targets.unsqueeze(0))

        soft_pseudo_label = torch.softmax(t_logits_u.detach(), dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(0.90).float()
        t_loss_u = torch.mean(
            -(soft_pseudo_label * torch.log_softmax(t_logits_a, dim=-1)).sum(dim=-1) * mask
        )
        weight_u = 1 * min(1., (step+1)/1) # lambda-u, uda_step
        t_loss_uda = t_loss_l + weight_u * t_loss_u

        s_input_ids = torch.tensor([l_input, a_input]).to(device)
        s_logits = student(s_input_ids)
        s_logits_l, s_logits_a = s_logits

        s_loss_l_old = F.cross_entropy(s_logits_l.unsqueeze(0).detach(), targets.unsqueeze(0))
        s_loss = criterion(s_logits_a.unsqueeze(0), hard_pseudo_label.unsqueeze(0))

        s_loss.backward()
        optimizer_s.step()

        with torch.no_grad():
            s_logits_l = student(torch.tensor(l_input).unsqueeze(0).to(device))
        s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets.unsqueeze(0))

        dot_product = s_loss_l_old - s_loss_l_new
        _, hard_pseudo_label = torch.max(t_logits_a.detach(), dim=-1)
        t_loss_mpl = dot_product * F.cross_entropy(t_logits_a.unsqueeze(0), hard_pseudo_label.unsqueeze(0))
        
        t_loss = t_loss_uda + t_loss_mpl

        t_loss.backward()
        optimizer_t.step()

        teacher.zero_grad()
        student.zero_grad()
    
        # step마다 Evalution 진행
        if step % 500 == 0:
            student.eval()
            correct, loss = 0, 0
            zero, one = 0, 0
            prediction = []
            with torch.no_grad():
                for _, batch in tqdm(enumerate(eval_dataloader)):
                    data = batch['input_ids'].cuda()
                    labels = batch['label']
                    output = student(data)
                    predicted = torch.max(output,1)[1]
                    
                    prediction += predicted.tolist()
                    zero += predicted.tolist().count(0)
                    one += predicted.tolist().count(1)
                    
                    correct += (predicted==labels.cuda()).sum()
                    loss += F.cross_entropy(output, labels.cuda()).item()

            eval_f1 = f1_score(eval_labels, prediction, average='macro')
            print(f'Epoch: {step+1} | Train Loss : {loss/len(eval_dataloader):.5f} | Test Acc : {correct/len(eval_dataset):.5f} | Zero : {zero} | One : {one} | F1 : {eval_f1:.5f}')
            # mlflow.log_metric('eval loss', loss/len(eval_dataloader))
            # mlflow.log_metric('Acc', correct.item()/len(eval_dataset))

            eval_acc = correct/len(eval_dataloader)
            if eval_acc > best_acc:
                torch.save(student.state_dict(), f'./save/meta_pseudo/result.pt')
                # mlflow.pytorch.log_model(student, 'model', registered_model_name="ToxicityText")
                best_acc = eval_acc


if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    # load tokenizer
    tokenizer = BertWordPieceTokenizer('./vocab_3.txt', lowercase=False)
    
    train(tokenizer, device)
