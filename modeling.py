from typing import Dict, List, Type, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, max_seq, embedding_dim, kernel, num_class) -> None:
        """Initialization."""
        """문맥을 보면서 혐오표현을 감지? 아니면 주위 단어를 보면서 감지?"""
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        
        self.conv1 = nn.Conv1d(
            in_channels=max_seq,
            out_channels=kernel,
            kernel_size=3)
        self.conv2 = nn.Conv1d(
            in_channels=max_seq,
            out_channels=kernel,
            kernel_size=4)
        self.conv3 = nn.Conv1d(
            in_channels=max_seq,
            out_channels=kernel,
            kernel_size=5)
        
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Linear(
            in_features=kernel*3,
            out_features=num_class)
        
        self.activation = nn.ReLU()
        
        self.softmax = nn.Softmax()


    def forward(self, input):
        """Forward."""
        # input size = (batch, max_seq)

        output = self.embedding(input) # size = (batch, max_seq, embedding_dim)
        
        output1 = self.conv1(output) # size = (batch, kernel, max_seq')
        output2 = self.conv2(output) # size = (batch, kernel, max_seq'')
        output3 = self.conv3(output) # size = (batch, kernel, max_seq''')
        
        output1 = self.activation(output1)
        output2 = self.activation(output2)
        output3 = self.activation(output3)

        m1 = nn.MaxPool1d(output1.size()[-1]) # batch
        m2 = nn.MaxPool1d(output2.size()[-1])
        m3 = nn.MaxPool1d(output3.size()[-1])
        output1 = m1(output1).view(input.size(0), -1) # size = (batch, kernel)
        output2 = m2(output2).view(input.size(0), -1) # size = (batch, kernel)
        output3 = m3(output3).view(input.size(0), -1) # size = (batch, kernel)

        output_cat = torch.cat((output1, output2, output3), 1) # size = (batch, kernel * 3)

        output = self.fc(self.dropout(output_cat))
        return output


if __name__ == "__main__":
    model = Model(vocab_size=50000, max_seq=200, embedding_dim=128, kernel=256, num_class=2)
    model.to(torch.device('cpu'))
    print(model)

    input = torch.randint(low=0, high=50000, size=(16,200)).to(torch.device('cpu'))
    print('input', input.shape)
    out = model(input)
    print(out.shape)
    print(out)