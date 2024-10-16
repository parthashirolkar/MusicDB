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
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, seq_len/2)
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, seq_len/4)
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # (batch_size, 256, seq_len/8)
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),  # (batch_size, 512, seq_len/16)
            nn.LeakyReLU(inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 256, seq_len/8)
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 128, seq_len/4)
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 64, seq_len/2)
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 1, seq_len)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
