import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def foward(self, input):
        embeddeding = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embeddeding)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def foward(self, input, en_hidden, en_cell):
        input = input.unsqueeze(0)
        embedding = self.dropout(self.embedding(input))

        output, (hidden, cell) = self.lstm(embedding, (en_hidden, en_cell))
        pred = self.fc(output.squeeze(0))

        return pred, hidden, cell


class Seq2Seq(BaseModel):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers, dropout)

    def forward(self, input, target, teacher_forcing=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.output_dim

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(input)

        input = target[0, :]
        for i in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[i] = output
            teacher_force = random.random() < teacher_forcing
            top1 = output.argmax(1)

            input = target[i] if teacher_force else top1

        return outputs
