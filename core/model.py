import torch
import torch.nn as nn

#==============================================================================#
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, bidirectoinal, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectoinal

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=bidirectoinal,
                            dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        #src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]

        _, (hidden, _) = self.lstm(embedded)
        #hidden = [n layers * n directions, batch size, hid dim]

        #hidden [-2, :, :] is the last of the forwards RNN if bidirectional
        #hidden [-1, :, :] is the last of the backwards RNN if bidirectional

        combined_hidden = (
            torch.cat(
                (hidden[-2,:,:], hidden[-1,:,:]),
                dim=1,
            ) if self.bidirectional
            else hidden[-1,:,:]
        )
        #combined_hidden = [batch size, hid dim * n directions]

        return combined_hidden


class SeqClassifier(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, class_num,
                 n_layers=2, bidirectional=False, dropout=0.5):
        super().__init__()

        self.name = "SeqClassifier"

        self.enc = Encoder(input_dim, emb_dim, hid_dim,
                           n_layers, bidirectional, dropout)

        self.fc = nn.Linear(
            hid_dim*2 if bidirectional else hid_dim,
            class_num,
        )

    def forward(self, src):
        #src = [batch size, src len]
        hidden = self.enc(src)
        #hidden = [batch size, hid dim * n directions]

        output = self.fc(hidden)
        #output = [batch size, class num]

        return output