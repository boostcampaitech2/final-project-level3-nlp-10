from typing import Dict, List, Type, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size : int, max_seq : int, embedding_dim : int, channel : int, num_class : int, device) -> None:
        """Filtering Model based CNN"""
        """문맥을 보면서 혐오표현을 감지? 아니면 주위 단어를 보면서 감지?"""
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self._init_weights(self.embedding)
        
        self.conv1 = nn.Conv1d(
            in_channels=max_seq,
            out_channels=channel,
            kernel_size=3)
        self._init_weights(self.conv1)
        self.conv2 = nn.Conv1d(
            in_channels=max_seq,
            out_channels=channel,
            kernel_size=4)
        self._init_weights(self.conv2)
        self.conv3 = nn.Conv1d(
            in_channels=max_seq,
            out_channels=channel,
            kernel_size=5)
        self._init_weights(self.conv3)
        
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Linear(
            in_features=channel * 3,
            out_features=num_class)
        self._init_weights(self.fc)
        
        self.activation = nn.ReLU()


    def forward(self, input):
        """Forward."""
        # input size = (batch, max_seq)

        output = self.embedding(input) # size = (batch, max_seq, embedding_dim)
        
        output1 = self.conv1(output) # size = (batch, channel, embedding_dim')
        output2 = self.conv2(output) # size = (batch, channel, embedding_dim'')
        output3 = self.conv3(output) # size = (batch, channel, embedding_dim''')
        
        # layer normalization and activation
        lm1 = nn.LayerNorm(output1.size()[-1], device=self.device)
        lm2 = nn.LayerNorm(output2.size()[-1], device=self.device)
        lm3 = nn.LayerNorm(output3.size()[-1], device=self.device)
        self._init_weights(lm1)
        self._init_weights(lm2)
        self._init_weights(lm3)
        output1 = self.activation(lm1(output1))
        output2 = self.activation(lm2(output2))
        output3 = self.activation(lm3(output3))

        m1 = nn.MaxPool1d(output1.size()[-1]) # batch
        m2 = nn.MaxPool1d(output2.size()[-1])
        m3 = nn.MaxPool1d(output3.size()[-1])
        output1 = m1(output1).view(input.size(0), -1) # size = (batch, channel)
        output2 = m2(output2).view(input.size(0), -1) # size = (batch, channel)
        output3 = m3(output3).view(input.size(0), -1) # size = (batch, channel)

        output_cat = torch.cat((output1, output2, output3), 1) # size = (batch, channel * 3)

        output = self.fc(self.dropout(output_cat))
        return output

    
    def _init_weights(self, module):
        """Initialize the weights with bert"""
        if isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=0.2)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)       


if __name__ == "__main__":
    model = Model(vocab_size=50000, max_seq=200, embedding_dim=256, channel=512, num_class=2, device=torch.device('cpu'))
    model.to(torch.device('cpu'))
    print(model)

    # batch_size = 16, max_seq_length = 200
    input = torch.randint(low=0, high=50000, size=(16,200)).to(torch.device('cpu'))
    print('input', input.shape)
    
    out = model(input)
    print(out.shape)
    # print(out)