from nltk.corpus import stopwords
import string
import os
from sklearn.utils import shuffle
from torch.distributions import uniform
import torch
import numpy as np
import gzip
import pickle

def un_gz(file_name):
    
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()



def preprocess_data(lines):
    """
    Split tokens on white space.
    Remove all punctuation from words.
    Remove all words that are not purely comprised of alphabetical characters.
    Remove all words that are known stop words.
    Remove all words that have a length <= 1 character.
    """
    formatted_lines = []
    for line in lines:
        # split into tokens by white space
        tokens = line.split()
        # remove punctuation from each token and convert to lowercase
        # table = str.maketrans('', '', string.punctuation)
        # tokens = [w.translate(table).lower() for w in tokens]
        tokens = [w.lower() for w in tokens]
        # # remove remaining tokens that are not alphabetic
        # tokens = [word for word in tokens if word.isalpha()]
        # # filter out stop words
        # stop_words = set(stopwords.words('english'))
        # tokens = [w for w in tokens if w not in stop_words]
        # # filter out short tokens
        # tokens = [word for word in tokens if len(word) > 1]
        # join the new tokens to form the formatted sentence
        sentence = " ".join(tokens)
        formatted_lines.append(sentence)
    return formatted_lines


def load_data(dataset_path):
    """Load test, validation and train data with their labels"""
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    with open(os.path.join(dataset_path, "train.txt"), 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_train.append(sentence)
            y_train.append(label)
    with open(os.path.join(dataset_path, "dev.txt"), 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_val.append(sentence)
            y_val.append(label)
    with open(os.path.join(dataset_path, "test.txt"), 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[1:].strip()
            x_test.append(sentence)
            y_test.append(label)

    return x_train, x_val, x_test, y_train, y_val, y_test

def _load_word_vec(path, word2idx=None):
    embedding2index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lexicons = line.split()
            word = lexicons[0]
            if word2idx is None or word in word2idx.keys():
                try:
                    coefs = torch.from_numpy(np.asarray(lexicons[1:], dtype='float32'))
                except:
                    pass
            embedding2index[word] = coefs
    return embedding2index

def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = 'embed/glove.840B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
            else:
                embedding_matrix[i] = uniform.Uniform(-0.01, 0.01).sample(torch.Size([300]))
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

def load_embedding(embed_path):
    """Load word embedding and return word-embedding vocabulary"""
    embedding2index = {}
    with open(embed_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lexicons = line.split()
            word = lexicons[0]
            try:
                coefs = torch.from_numpy(np.asarray(lexicons[1:], dtype='float32'))
            except:
                pass
            embedding2index[word] = coefs
    return embedding2index


def load_embedding_matrix(embedding, words, embedding_size):
    """Add new words in the embedding matrix and return it"""
    embedding_matrix = torch.zeros(len(words), embedding_size)
    for i, word in enumerate(words):
        # Note: PAD embedded as sequence of zeros
        if word not in embedding:
            # if word != 'PAD':
            embedding_matrix[i] = uniform.Uniform(-0.25, 0.25).sample(torch.Size([embedding_size]))
        else:
            embedding_matrix[i] = embedding[word]
    return embedding_matrix


def get_loaders(x_train, x_val, x_test, y_train, y_val, y_test, batch_size, device):
    """Return iterables over train, validation and test dataset"""

    # convert labels to vectors and put on device
    y_train = torch.from_numpy(np.asarray(y_train, dtype='int32')).to(device)
    y_val = torch.from_numpy(np.asarray(y_val, dtype='int32')).to(device)
    y_test = torch.from_numpy(np.asarray(y_test, dtype='int32')).to(device)

    # convert sequences of indexes to tensors and put on device
    x_train = torch.from_numpy(np.asarray(x_train, dtype='int32')).to(device)
    x_val = torch.from_numpy(np.asarray(x_val, dtype='int32')).to(device)
    x_test = torch.from_numpy(np.asarray(x_test, dtype='int32')).to(device)

    train_array = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_array, batch_size, shuffle=True)

    val_array = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_array, batch_size)

    test_array = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_array, batch_size)

    return train_loader, val_loader, test_loader
