import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim_1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim_1 * 2, hidden_dim_2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim_2, hidden_dim_1):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(hidden_dim_2 * 2, hidden_dim_2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim_2 * 2, hidden_dim_1, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x

class BiLSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2):
        super(BiLSTMSeq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim_1, hidden_dim_2)
        self.decoder = Decoder(hidden_dim_2, hidden_dim_1)
        self.output_layer = nn.Linear(hidden_dim_1 * 2, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        output = self.output_layer(x)
        return output
