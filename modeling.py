"""
    CNN-LSTM 모델
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, vocab_size : int, embedding_dim : int, channel : int, num_class : int, dropout1: float, dropout2: float) -> None:
        """Filtering Model based CNN"""
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self._init_weights(self.embedding)
        
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=channel,
            kernel_size=2)
        self._init_weights(self.conv1)
        self.conv2 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=channel,
            kernel_size=4)
        self._init_weights(self.conv2)
        self.conv3 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=channel,
            kernel_size=6)
        self._init_weights(self.conv3)

        self.classifier = nn.Linear(
            in_features=channel * 5,
            out_features=num_class)
        self._init_weights(self.classifier)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=channel,
            batch_first=True,
            bidirectional=True,
            dropout=0.2)
        self._init_weights(self.lstm)
        
        self.dropout = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        
        self.relu = nn.ReLU()

    def forward(self, input):
        """Forward."""
        # input size = (batch, max_seq)
        # MLflow에서 강제로 float으로 형변환하기 때문에 float인경우 long타입으로 변환해야 합니다.
        if type(input.tolist()[0][0])==float:
            input = torch.tensor(input.tolist(),dtype=torch.long)

        embd = self.embedding(input) # size = (batch, max_seq, embedding_dim)
        embd = self.dropout(embd)
        output = torch.transpose(embd, 1, 2) # size = (batch, embedding_dim, max_seq)
        
        lstm_output, (hn, cn) = self.lstm(embd) # size = (batch, max_seq, channel * 2)
        
        lstm_output = lstm_output[:,-1,:] # size = (batch, channel * 2)

        output1 = self.conv1(output) # size = (batch, embedding_dim, max_seq')
        output2 = self.conv2(output) # size = (batch, embedding_dim, max_seq'')
        output3 = self.conv3(output) # size = (batch, embedding_dim, max_seq''')
        
        # layer normalization and activation
        device = torch.device('cuda') if self.embedding.weight.is_cuda else torch.device('cpu') # model의 위치(cpu or gpu)를 판단
        lm1 = nn.LayerNorm(output1.size()[-1]).to(device)
        lm2 = nn.LayerNorm(output2.size()[-1]).to(device)
        lm3 = nn.LayerNorm(output3.size()[-1]).to(device)
        self._init_weights(lm1)
        self._init_weights(lm2)
        self._init_weights(lm3)
        output1 = self.dropout2(self.relu(lm1(output1))) # size = (batch, channel, embedding_dim)
        output2 = self.dropout2(self.relu(lm2(output2)))
        output3 = self.dropout2(self.relu(lm3(output3)))
        
        m1 = nn.MaxPool1d(output1.size(-1))
        m2 = nn.MaxPool1d(output2.size(-1))
        m3 = nn.MaxPool1d(output3.size(-1))
        output1 = m1(output1).view(output1.size(0),-1) # size = (batch, channel)
        output2 = m2(output2).view(output1.size(0),-1) # size = (batch, channel)
        output3 = m3(output3).view(output1.size(0),-1) # size = (batch, channel)

        output_cat = torch.concat((output1, output2, output3, lstm_output), 1) # size = (batch, channel * 5)

        output = self.dropout(self.relu(output_cat))

        return self.classifier(output)


    def _init_weights(self, module):
        """
            허깅페이스에서 사용하는 initialization 코드를 적용합니다. 
            torch에서 각 모듈에 대해 다른 initialization을 같은 분포로 해주는 역할을 합니다.
            Reference : https://github.com/huggingface
        """
        if isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=0.02)
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
    """
        Test code
        모델이 주어진 input에 정상적으로 돌아가는지 확인하는 코드입니다.
    """
    print('='*25, 'Test', '='*25)
    model = Model(vocab_size=50000, embedding_dim=8, channel=16, num_class=2, dropout1=0.1, dropout2=0.2)
    model.to(torch.device('cpu'))
    print(model)

    # batch_size = 1, max_seq_length = 200
    batch_size = 1
    input = torch.randint(low=0, high=50000, size=(batch_size, 200)).to(torch.device('cpu'))
    print('input', input.shape)
    
    out = model(input)
    print(out.shape)
    assert(out.size(0)==batch_size and out.size(1)==2), 'output.shape is not match (batch_size, num_class)'
