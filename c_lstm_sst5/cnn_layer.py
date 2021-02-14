import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data

class CNN(nn.Module):
    """Convolutional model"""
    def __init__(self, embedding_matrix, embedding_dim, n_filters=150, embedding_dropout=0.5,vocab_size=None):
        super().__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        else:
            if vocab_size is None:
                raise Exception("Could not create new embedding: missing vocabulary size!")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv_layers = nn.Conv2d(in_channels=1,out_channels=n_filters,kernel_size = [3,embedding_dim], bias=True)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, x):
        # text = [batch size, sent len]
        embedded = self.embedding_dropout(self.embedding(x))
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv_layers(embedded)).squeeze(3)
        # conved_n = [batch size, n_filters, sent len - filter_size + 1]
        conved = conved.permute(0,2,1)
        # conved_n = [batch size, sent len - filter_size + 1, n_filters]
        return conved


if __name__ == '__main__':
    # Testing Model
    print("Testing CNN model...")
    x = torch.ones(4, 10).long()
    sentence = ["This is just a test to check whether the model work"]
    # embedding = nn.Embedding(11, 10)
    model = CNN(None, 10, 150, 0.5, 11)
    out = model(x)
    print(out.shape)
    print("Test passed!")