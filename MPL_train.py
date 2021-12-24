"""
    Meta Pseudo Labeling이 구현된 코드입니다.
    Reference: https://github.com/kekmodel/MPL-pytorch
"""
import os
import gc
import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from transformers import ElectraForSequenceClassification
from tokenizers import BertWordPieceTokenizer

import modeling
from utils import Config, set_seed, GOOGLE_APPLICATION_CREDENTIAL, MLFLOW_TRACKING_URI
from data import load_dataset, punctuation, punctuation2, tokenized_dataset
from tqdm import trange, tqdm

# MLFlow 추적을 위한 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=GOOGLE_APPLICATION_CREDENTIAL
os.environ['MLFLOW_TRACKING_URI']=MLFLOW_TRACKING_URI

set_seed(42)

config = Config(
    dropout1=0.3,
    dropout2=0.4,
    learning_rate=1e-3,
    label_smoothing=0.0,
    epochs=50,
    embedding_dim=100,
    channel=128)


def seed_init_fn(x):
   seed = 42 + x
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(tokenizer, device) -> None:
    # Print Hyperparameters
    print(f'config : {config.__dict__}')
    # mlflow.log_params(config.__dict__)

    # Train Dataset
    df = pd.read_csv('labeled.csv')
    p_df = pd.read_csv('twitch.csv')
    eval_df = pd.read_csv('test2.csv')

    # pseudo labeling할 데이터 중 2.8만개를 샘플로 사용합니다.
    true_label = p_df[(p_df['none']<p_df['curse'])==True]
    false_label = p_df.drop(true_label.index, axis=0).reset_index().drop(['index'], axis=1)
    false_label = false_label.sample(frac=0.025, random_state=42)
    true_label = true_label.sample(frac=0.25)
    true_label = true_label.append(false_label)
    true_label = true_label.sample(frac=1, random_state=42).reset_index().drop(['index'], axis=1)
    p_df = true_label

    # weak augmentation
    p_df = punctuation(p_df)
    
    # strong augmentation
    a_df = punctuation2(p_df['text'])

    labels = list(df['label'])
    eval_labels = list(eval_df['label'])
    print(f'Test labels 0 : {eval_labels.count(0)}, 1 : {eval_labels.count(1)}')

    df = tokenized_dataset(tokenizer, df)
    p_df = tokenized_dataset(tokenizer, p_df)
    a_df = tokenized_dataset(tokenizer, a_df)
    eval_df = tokenized_dataset(tokenizer, eval_df)
    
    dataset = load_dataset(df, labels)
    p_dataset = load_dataset(p_df)
    a_dataset = load_dataset(a_df)
    eval_dataset = load_dataset(eval_df, eval_labels)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_init_fn)
    p_dataloader = DataLoader(p_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_init_fn)
    a_dataloader = DataLoader(a_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_init_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    

    # Load teacher model(pretrained), studentmodel
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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer_s = torch.optim.SGD(student.parameters(), lr=1e-7)
    optimizer_t = torch.optim.SGD(teacher.parameters(), lr=1e-7)
    scaler_s = torch.cuda.amp.GradScaler()
    scaler_t = torch.cuda.amp.GradScaler()
    scheduler_t = get_cosine_schedule_with_warmup(
        optimizer=optimizer_t, num_warmup_steps=0, num_training_steps=len(p_dataloader))
    scheduler_s = get_cosine_schedule_with_warmup(
        optimizer=optimizer_s, num_warmup_steps=0, num_training_steps=len(p_dataloader))


    # Meta Pseudo Labeling
    print('------Start Training------')

    torch.cuda.empty_cache()
    gc.collect()

    best_f1 = 0
    prev_f1, patient = -1, 0
    for step in trange(len(p_dataloader)):
        teacher.train()
        student.train()

        labeled = next(iter(dataloader))
        unlabeled = next(iter(p_dataloader))
        a_labeled = next(iter(a_dataloader))

        # labeled data
        l_input = labeled['input_ids']
        l_attention_mask = labeled['attention_mask']
        targets = labeled['label'].to(device)

        # reference의 strong augmentation을 augmentation한 unlabeled dataset으로 가정
        a_input = a_labeled['input_ids']
        a_attention_mask = a_labeled['attention_mask']

        # reference의 weak augmentation을 unlabeled dataset으로 가정
        u_input = unlabeled['input_ids']
        u_attention_mask = unlabeled['attention_mask']
        
        with torch.cuda.amp.autocast():
            # teacher model에 먹일 input 구성 (labeled, augmention, unlabeled)
            t_input_ids = torch.cat((l_input, a_input, u_input)).to(device)
            t_attention_mask = torch.cat((l_attention_mask, a_attention_mask, u_attention_mask)).to(device)

            t_logits = teacher(input_ids=t_input_ids, attention_mask=t_attention_mask)['logits']

            t_logits_l = t_logits[:batch_size]
            t_logits_a, t_logits_u = t_logits[batch_size:].chunk(2)
        
            # teacher모델의 labeled data에 대한 loss
            t_loss_l = criterion(t_logits_l, targets)

            # augmentation을 통한 data의 label과 unlabeled data의 로스의 비교(unlabeled data로부터 증강되었기 때문)
            soft_pseudo_label = torch.softmax(t_logits_u.detach()/0.9, dim=-1) # temperature
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(0.60).float()
            t_loss_u = torch.mean( # KL.Div loss
                -(soft_pseudo_label * torch.log_softmax(t_logits_a, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = 1 * min(1., (step+1)/1) # lambda-u, uda_step
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            # student model에 먹일 input 구성 (labeled, augmention)
            s_input_ids = torch.cat((l_input, a_input)).to(device)

            s_logits = student(s_input_ids)
            s_logits = F.sigmoid(s_logits)
            s_logits_l, s_logits_a = s_logits[:batch_size], s_logits[batch_size:]

            # 업데이트 되지 않은 student 모델의 labeled data에 대한 로스값(labeled data에 대한 validation)
            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            
            # augmented data에 대해서 student가 학습
            s_loss = criterion(s_logits_a, hard_pseudo_label)

        scaler_s.scale(s_loss).backward()
        scaler_s.step(optimizer_s)
        scaler_s.update()
        scheduler_s.step()

        with torch.cuda.amp.autocast():
            # 업데이트 된 student 모델의 labeled data에 대한 로스
            with torch.no_grad():
                s_logits_l = student(l_input.to(device))
                s_logits_l = F.sigmoid(s_logits_l)
                s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
            
            # teacher coefficient : https://github.com/kekmodel/MPL-pytorch/issues/6
            dot_product = s_loss_l_old - s_loss_l_new

            # compute the teacher's gradient from student's feedback
            _, hard_pseudo_label = torch.max(t_logits_a.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_a, hard_pseudo_label)
            
            t_loss = t_loss_uda + t_loss_mpl # t_loss_uda = t_loss_l + t_loss_unlabeled

        scaler_t.scale(t_loss).backward()
        scaler_t.step(optimizer_t)
        scaler_t.update()
        scheduler_t.step()

        teacher.zero_grad()
        student.zero_grad()
    
        # step마다 Evalution 진행
        if step > 0 and step % 10 == 0:
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

            if eval_f1 > best_f1:
                # 가장 좋을 때의 모델을 저장합니다.
                torch.save(student.state_dict(), f'./save/meta_pseudo/result_temp.pt')
                best_f1 = eval_f1
        
            if prev_f1 == eval_f1:
                patient += 1
                if patient == 20:
                    break
            else:
                patient = 0
            prev_f1 = eval_f1

    print(f'best f1 = {best_f1}')


def finetune(tokenizer, device):
    """
        MPL이 적용된 student 모델을 다시 labeled data로 Fine tuning합니다.
    """
    # Load datasets
    df = pd.read_csv('labeled.csv')
    eval_df = pd.read_csv('test2.csv')
    
    labels = list(df['label'])
    eval_labels = list(eval_df['label'])

    df = punctuation(df)
    
    df = tokenized_dataset(tokenizer, df)
    eval_df = tokenized_dataset(tokenizer, eval_df)
    
    dataset = load_dataset(df, labels)
    eval_dataset = load_dataset(eval_df, eval_labels)
    
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    epochs = 10
    model = modeling.Model(
        vocab_size=30000,
        embedding_dim=100,
        channel=128,
        num_class=2,
        dropout1=0.3,
        dropout2=0.4
    )
    model.load_state_dict(torch.load('./save/meta_pseudo/result_temp.pt'))
    model.to(device)

    # Set criterion, optimizer, scheduler
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs = epochs,
        max_lr=0.01,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,
    )
    
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for i, labeled in enumerate(dataloader):
            input = labeled['input_ids'].to(device)
            label = labeled['label'].to(device)

            output = model(input)
            loss = criterion(output, label)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        with torch.no_grad():
            # Evalutaion
            model.eval()
            correct = 0
            prediction = []
            for j, batch in enumerate(eval_dataloader):
                input = batch['input_ids'].cuda()
                label = batch['label'].cuda()
                
                output = model(input)
                preds = output.argmax(-1)
                prediction += preds.tolist()
                correct += (preds==label).sum().item()
            
            eval_acc = correct/len(eval_dataset)
            f1 = f1_score(eval_labels, prediction, average='macro')

        print(f'Epoch: {epoch+1} | Train Loss : {running_loss/len(dataloader):.5f} | Acc : {eval_acc:.5f} | F1 : {f1:.3f}')


if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    # load tokenizer
    tokenizer = BertWordPieceTokenizer('./vocab_3.txt', lowercase=False)
    
    train(tokenizer, device)
    # finetune(tokenizer, device)
