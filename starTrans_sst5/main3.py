from typing import Iterator, Text

from torch import mode
import torch.nn.functional as F
from cls_model import *
from train import *
from fastNLP.models import STSeqCls
import torch
from torchtext import data
import os

save_path = "model"

batch_size = 32
epoch = 10
weight_decay = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Detected device:", device)

if not os.path.exists(save_path):
    os.makedirs(save_path)

TEXT = data.Field(tokenize=lambda x: x.split(), batch_first=True, lower=True)
LABEL = data.LabelField(dtype=torch.float)

def get_dataset(corpur_path, text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]             #torchtext中于文件配对关系
    examples = []

    with open(corpur_path) as f:
        for line in f.readlines():
            label = int(line[0])
            text = line[1:].strip()
            examples.append(data.Example.fromlist([text, label], fields))
    
    return examples, fields

train_examples, train_fields = get_dataset("dataset/sst5/train.txt", TEXT, LABEL)
dev_examples, dev_fields = get_dataset("dataset/sst5/dev.txt", TEXT, LABEL)
test_examples, test_fields = get_dataset("dataset/sst5/test.txt", TEXT, LABEL)


train_data = data.Dataset(train_examples, train_fields)
dev_data = data.Dataset(dev_examples, dev_fields)
test_data = data.Dataset(test_examples, test_fields)

print('len of train data:', len(train_data))              #1000
print('len of dev data:', len(dev_data))                  #200
print('len of test data:', len(test_data))


TEXT.build_vocab(train_data, vectors='glove.840B.300d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(LABEL.vocab)

train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits((train_data, dev_data, test_data), batch_size=batch_size, sort=False, device=device)

embedding_matrix = TEXT.vocab.vectors
print(embedding_matrix)

print("Building model...")
model = STSeqCls(embedding_matrix, 5)
model.to(device)


def train(model, iterator, optimizer, device):
    model.train()
    tot_loss = 0.0
    tot_accuracy = 0.0
    for batch in tqdm.tqdm(iterator):
        batch.text = batch.text.long()
        batch.label = batch.label.long()
        mask = 1 - (batch.text == TEXT.vocab.stoi['<pad>']).float()
        batch.text.to(device)
        mask.to(device)

        out = model(batch.text, mask)
        out = F.softmax(out, dim=1)
        print(out)

        loss = F.cross_entropy(out, batch.label.to(device), reduction='sum')
        tot_loss += loss.item()

        predictions = torch.max(out, 1)[1]
        print(predictions)
        tot_accuracy += torch.sum(predictions == batch.label).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = tot_loss / len(train_data)
    acc = tot_accuracy / len(train_data)
    return loss, acc

def evaluate(model, iterator, device, mode):
    model.eval()
    with torch.no_grad():
        tot_loss = 0.0
        tot_accuracy = 0.0
        for batch in tqdm.tqdm(iterator):
            batch.text = batch.text.long()
            batch.label = batch.label.long()
            print(batch.label)
            batch.text.to(device)
            mask = 1 - (batch.text == TEXT.vocab.stoi['<pad>']).float()
            mask.to(device)
            
            out = model(batch.text, mask)
            out = F.softmax(out, dim=1)

            tot_loss += F.cross_entropy(out, batch.label.to(device), reduction='sum').item()

            predictions = torch.max(out, 1)[1]
            tot_accuracy += torch.sum(predictions == batch.label).item()
        if mode == 'dev':
            loss = tot_loss / len(dev_data)
            acc = tot_accuracy / len(dev_data)
        else:
            loss = tot_loss / len(test_data)
            acc = tot_accuracy / len(test_data)
        
        return loss, acc


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
best_val_acc = 0.0
print("Start training")
for epoch in range(epoch):
    
    print("Epoch {}".format(epoch+1))
    train_loss, train_acc = train(model, train_iterator, optimizer, device)
    val_loss, val_acc = evaluate(model, dev_iterator, device)
    print("\tTrain Loss {:.3f} | Train Acc {:.3f}".format(train_loss, train_acc))
    print("\t Val Loss {:.3f} | Test Acc {:.3f}".format(val_loss, val_acc))
    if val_acc > best_val_acc:
        print("Saving model...")
        torch.save(model.state_dict(), 'model/trained.pth')
        print("Model saved in: " + save_path)

print("Start testing")
model.load_state_dict(torch.load('model/trained.pth'))
_, test_acc = evaluate(model, test_iterator, device)
print('Test acc: {:.3f}'.format(test_acc))
with open("result.txt", 'a+') as f:
    f.write(str(test_acc))
    f.write('\n')


