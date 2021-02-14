from nltk.util import print_string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from dynamic_rnn import DynamicLSTM

class RNN(nn.Module):
    """LSTM model"""
    def __init__(self, embedding_matrix, vocab_size=None,embedding_dim=300, hidden_dim=150, dropout=0.5, embedding_dropout=0.5, output_dim=2):
        super(RNN, self).__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        else:
            if vocab_size is None:
                raise Exception("Could not create new embedding: missing vocabulary size!")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm = DynamicLSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(300, 128),
                                nn.Dropout(0.8),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, 2),
                                nn.Softmax())

        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input: [batch size, len]
        seq_len = torch.sum(input != 0, dim=-1)

        embedd = self.embedding_dropout(self.embedding(input))

        # x_sort_idx = torch.sort(seq_len, descending=True)[1].long()
        # seq_len = seq_len[x_sort_idx]
        # embedd = embedd[x_sort_idx]

        # embedd_p = torch.nn.utils.rnn.pack_padded_sequence(embedd, seq_len, batch_first=True)
        out, _ = self.lstm(embedd, seq_len)
        # hidden = [num layers * num directions, batch size, hid dim]
        
        out = self.dropout(out)

        out = self.f1(out[:, -1, :])
        # hidden = [batch size, output_dim]

        final = self.f2(out)
        return final

if __name__ == '__main__':
    # Testing Model
    print("Testing RNN model...")
    x = torch.ones(5, 10).long()
    sentence = ["This is just a test to check whether the model work"]
    # embedding = nn.Embedding(11, 10)
    model = RNN(None,20)
    out = model(x)
    print(out.shape)
    print("Test passed!")