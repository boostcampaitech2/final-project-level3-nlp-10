import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import adaptive
from torch.nn.modules.pooling import AdaptiveAvgPool1d


class Model(nn.Module):
    def __init__(self, vocab_size : int, embedding_dim : int, channel : int, num_class : int, dropout1: float, dropout2: float, device) -> None:
        """Filtering Model based CNN"""
        super().__init__()
        self.device = device
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

        self.fc = nn.Linear(
            in_features=channel * 3,
            out_features=channel,
        )
        self._init_weights(self.fc)
        self.classifier = nn.Linear(
            in_features=channel * 4,
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
        self.sigmoid = nn.Sigmoid()

        self.AvgPool = nn.AdaptiveAvgPool1d(channel)


    def forward(self, input):
        """Forward."""
        # input size = (batch, max_seq)

        embd = self.embedding(input) # size = (batch, max_seq, embedding_dim)
        embd = self.dropout(embd)
        output = torch.transpose(embd, 1, 2) # size = (batch, embedding_dim, max_seq)
        
        lstm_output, (hn, cn) = self.lstm(embd) # size = (batch, max_seq, channel * 2)
        
        lstm_output = torch.transpose(lstm_output, 0, 1)[-1] # size = (batch, channel * 2)
        lstm_output = self.AvgPool(lstm_output)
        
        output1 = self.conv1(output) # size = (batch, embedding_dim, max_seq')
        output2 = self.conv2(output) # size = (batch, embedding_dim, max_seq'')
        output3 = self.conv3(output) # size = (batch, embedding_dim, max_seq''')
        
        # layer normalization and activation
        lm1 = nn.LayerNorm(output1.size()[-1], device=self.device)
        lm2 = nn.LayerNorm(output2.size()[-1], device=self.device)
        lm3 = nn.LayerNorm(output3.size()[-1], device=self.device)
        self._init_weights(lm1)
        self._init_weights(lm2)
        self._init_weights(lm3)
        output1 = self.dropout2(self.relu(lm1(output1))) # size = (batch, channel, embedding_dim)
        output2 = self.dropout2(self.relu(lm2(output2)))
        output3 = self.dropout2(self.relu(lm3(output3)))

        m1 = nn.MaxPool1d(output1.size(-1))
        m2 = nn.MaxPool1d(output2.size(-1))
        m3 = nn.MaxPool1d(output3.size(-1))
        output1 = m1(output1).view(input.size(0), -1) # size = (batch, channel)
        output2 = m2(output2).view(input.size(0), -1) # size = (batch, channel)
        output3 = m3(output3).view(input.size(0), -1) # size = (batch, channel)

        output_cat = torch.concat((output1, output2, output3, lstm_output), 1) # size = (batch, channel * 4)

        output = self.dropout(self.relu(output_cat))

        return self.classifier(output)


    def _init_weights(self, module):
        """Initialize the weights with bert"""
        """Reference : huggingface.co"""
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
    """Test"""
    print('='*25, 'Test', '='*25)
    model = Model(vocab_size=50000, embedding_dim=8, channel=16, num_class=2, dropout1=0.1, dropout2=0.2, device=torch.device('cpu'))
    model.to(torch.device('cpu'))
    print(model)

    # batch_size = 16, max_seq_length = 200
    batch_size = 16
    input = torch.randint(low=0, high=50000, size=(batch_size, 200)).to(torch.device('cpu'))
    print('input', input.shape)
    
    out = model(input)
    print(out.shape)
    assert(out.size(0)==batch_size and out.size(1)==2), 'output.shape is not match (batch_size, num_class)'