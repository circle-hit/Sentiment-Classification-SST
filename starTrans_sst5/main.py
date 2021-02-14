from cls_model import *
from data_helper import *
from vocabulary import *
from train import *
from fastNLP.models.star_transformer import STSeqCls
import torch
import os

dataset_path = "dataset/sst5"
embedding_path = "embed/glove.840B.300d.txt"
save_path = "model"
test_path = "dataset"

batch_size = 32
epoch = 15
weight_decay = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Detected device:", device)

if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Loading dataset...")
x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset_path)

print("Preprocessing dataset...")
x_train, x_val, x_test = preprocess_data(x_train), preprocess_data(x_val), preprocess_data(x_test)

print("Creating vocabulary...")
vocabulary = Voc()
vocabulary.addSentences(x_train)
vocabulary.addSentences(x_val)
vocabulary.addSentences(x_test)
print("Vocabulary contains " + str(vocabulary.num_words) + " words")

max_len = len(max(x_train, key=len))

# convert input sentences to their indices and pad them
x_train_idx = vocabulary.pad_sentences(x_train, max_len)
x_val_idx = vocabulary.pad_sentences(x_val, max_len)
x_test_idx = vocabulary.pad_sentences(x_test, max_len)

embedding_matrix = build_embedding_matrix(word2idx=vocabulary.word2index, embed_dim=300, dat_fname='{0}_{1}_embedding_matrix.dat'.format('dataset_path', str(300)))
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

# embedding_matrix = None
# if embedding_path is not None:
#     print("Loading embedding...")
#     # load embedding from path
#     embedding2index = load_embedding(embedding_path)
#     # get embedding matrix corresponding to the words in vocabulary
#     embedding_matrix = load_embedding_matrix(embedding2index, vocabulary.get_words(), 300)

print("Building model...")
model = STSeqCls(embedding_matrix, 5)
model.to(device)

x_train, x_val, x_test = get_loaders(x_train_idx, x_val_idx, x_test_idx, y_train, y_val, y_test, batch_size, device)

optimizer = torch.optim.Adam(model.parameters())

print("Start training")
for epoch in range(epoch):
    
    print("Epoch {}".format(epoch+1))
    train_loss, train_acc = train(model, x_train, optimizer, device)
    test_loss, test_acc = evaluate(model, x_test, device)
    print("\tTrain Loss {:.3f} | Train Acc {:.3f}".format(train_loss, train_acc))
    print("\tTest Loss {:.3f} | Test Acc {:.3f}".format(test_loss, test_acc))

print("Start testing")
_, test_acc = evaluate(model, x_test, device)
print('Test acc: {:.3f}'.format(test_acc))
with open("result.txt", 'a+') as f:
    f.write(str(test_acc))
    f.write('\n')

print("Saving model...")
torch.save(model.state_dict(), 'model/trained.pth')
print("Model saved in: " + save_path)

