import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from dynamic_rnn import DynamicLSTM

class C_LSTM(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim=300, n_filters=150, filter_size=3, embedding_dropout=0.5, hidden_dim=150, dropout=0.5, output_dim=5, vocab_size=None):
        super().__init__()

        # self.filter_sizes = filter_sizes
        # self.max_len = max_len

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        else:
            if vocab_size is None:
                raise Exception("Could not create new embedding: missing vocabulary size!")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.requires_grad = True

        # self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1,
        #                                   out_channels=n_filters,
        #                                   kernel_size=(fs, embedding_dim),
        #                                   bias=True) for fs in self.filter_sizes])
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, embedding_dim), bias=True)

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.lstm = nn.LSTM(n_filters, hidden_dim, num_layers=1, batch_first=True)
        # self.lstm = DynamicLSTM(n_filters, hidden_dim, only_use_last_hidden_state=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x = [batch size, sent len]
        embedded = self.embedding_dropout(self.embedding(x))
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        
        # max_feature_length = self.max_len - max(self.filter_sizes) + 1
        # conved_outputs = []
        # for i, _ in enumerate(self.filter_sizes):
        #     conved_n = F.relu(self.conv_layers[i](embedded)).squeeze()
        #     # conved_n = [batch size, n_filters, sent len - filter_size + 1]
        #     conved_n = conved_n[:, :, :max_feature_length]

        #     conved_n = conved_n.permute(0,2,1)
        #     # conved_n = [batch size, sent len - filter_size + 1, n_filters]

        #     conved_outputs.append(conved_n)
        # conved = cat(conved_outputs, dim=1)

        conved = F.relu(self.conv_layer(embedded)).squeeze()
        conved = conved.permute(0, 2, 1)

        
        output, (hidden, cell) = self.lstm(conved)
        # hidden = [num layers * num directions, batch size, hid dim]

        # hidden = hidden[-1]
        hidden = hidden.squeeze()
        # hidden = [batch size, hid dim]

        hidden = self.dropout(hidden)
        # hidden = [batch size, hid dim]

        hidden = F.softmax(self.fc(hidden), dim=-1)
        # hidden = [batch size, output_dim]

        return hidden

if __name__ == '__main__':
    # Testing Model
    torch.manual_seed(1)
    print("Testing C_LSTM model...")
    # x = torch.ones(4, 5).long()
    x = torch.tensor([[2,3,4,5,0],[3,4,0,0,0]])
    sentence = ["This is just a test to check whether the model work"]
    # embedding = nn.Embedding(11, 10)
    model = C_LSTM(None, vocab_size=10)
    out = model(x)
    print(out.shape)
    print("Test passed!")
