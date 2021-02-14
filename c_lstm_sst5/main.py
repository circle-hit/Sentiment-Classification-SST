from os import write
import torch
import yaml
from data_helper import *
from vocabulary import *
from c_lstm import *
from lstm_layer import RNN
from train import *
from reg import *
import os

params = yaml.safe_load(open('params.yml',encoding='utf-8'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Detected device:", device)

if not os.path.exists(params['save_path']):
    os.makedirs(params['save_path'])

print("Loading dataset...")
x_train, x_val, x_test, y_train, y_val, y_test = load_data(params['dataset_path'])

print("Preprocessing dataset...")
x_train, x_val, x_test = preprocess_data(x_train), preprocess_data(x_val), preprocess_data(x_test)

print("Creating vocabulary...")
vocabulary = Voc()
vocabulary.addSentences(x_train)
print("Vocabulary contains " + str(vocabulary.num_words) + " words")


# print(vocabulary.get_words())
# print(vocabulary.word2index)
# test_sen = ["this is a visually stunning rumination on love , memory , history and the war between art and commerce wonderful ."]
# test_sen_idx = vocabulary.pad_sentences(test_sen, 30, 'left')
# print(test_sen_idx)

max_len = 50

# convert input sentences to their indices and pad them
x_train_idx = vocabulary.pad_sentences(x_train, max_len, 'left')
x_val_idx = vocabulary.pad_sentences(x_val, max_len, 'left')
x_test_idx = vocabulary.pad_sentences(x_test, max_len, 'left')

embedding_matrix = build_embedding_matrix(word2idx=vocabulary.word2index, embed_dim=300, dat_fname='{0}_{1}_embedding_matrix.dat'.format(params['dataset_path'], str(300)))
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

# embedding_matrix = None
# if params['embedding_path'] is not None:
#     print("Loading embedding...")
#     # load embedding from path
#     embedding2index = load_embedding(params['embedding_path'])
#     # get embedding matrix corresponding to the words in vocabulary
#     embedding_matrix = load_embedding_matrix(embedding2index, vocabulary.get_words(), 300, params['save_embedding_path'])


print("Building model...")
model = C_LSTM(embedding_matrix)
model.to(device)

reg_loss = 0.0
if params['weight_decay'] > 0:
   reg_loss = Regularization(model, params['weight_decay'], p=2).to(device)
else:
   print("no regularization")

x_train, x_val, x_test = get_loaders(x_train_idx, x_val_idx, x_test_idx, y_train, y_val, y_test, params['batch_size'], device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_acc = 0.0

print("Start training")
for epoch in range(params['epoch']):
    print("Epoch {}".format(epoch+1))
    train_loss, train_acc = train(model, x_train, optimizer, device)
    val_loss, val_acc = evaluate(model, x_val, device)
    print("\tTrain Loss {:.3f} | Train Acc {:.3f}".format(train_loss, train_acc))
    print("\tTest Loss {:.3f} | Test Acc {:.3f}%".format(val_loss, val_acc))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print("Saving model...")
        torch.save(model.state_dict(), 'model/trained.pth')
        print("Model saved in: " + params['save_path'])

print("Start testing")
model.load_state_dict(torch.load('model/trained.pth'))
_, test_acc = evaluate(model, x_test, device)
print('Test acc: {:.3f}'.format(test_acc))
with open("result.txt", 'a+') as f:
    f.write(str(test_acc))
    f.write('\n')




