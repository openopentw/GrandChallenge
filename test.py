def use_device(device):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return
use_device('gpu')  # cpu / gpu

import json
import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dropout, Dense, Embedding, LSTM, GRU, Bidirectional, merge
from keras.layers.core import Reshape, Lambda
from keras.layers.merge import Add, Dot, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# parameter
ID = 2

print("\nID = {}\n".format(ID))
model_path = './model/model_{}.h5'.format(ID)
weights_path = './weights/weights_{}.weights'.format(ID)
word_index_path = './word_index/{}.json'.format(ID)
output_path = './subm/{}.csv'.format(ID)

# data_path = './preprocess/corpus.txt'
test_data_path = './preprocess/test_corpus.csv'
word_vec_path = './outside_data/wiki.zh.vector'
EMBD_DIM = 400
# word_vec_path = './outside_data/my.cbow.200d.txt'
# EMBD_DIM = 200
q_maxlen = 172
a_maxlen = 72

# load testing data
test_data = pd.read_csv(test_data_path).drop('id', axis=1)
test_data = test_data.values

# load tokenizer
tokenizer = Tokenizer()
with open(word_index_path) as f:
    word_index = json.load(f)
tokenizer.word_index = word_index
print('Found {} unique tokens.'.format(len(word_index)))

# generate sequences
original_shape = test_data.shape
test_data = test_data.reshape(test_data.size)
sequences = tokenizer.texts_to_sequences(test_data)
test_data = test_data.reshape(original_shape)

# map each questions and answers
q_sequences = []
a_sequences = []
for i,seq in enumerate(sequences):
    if i % 7 == 0:  # question
        for _ in range(6):
            q_sequences += [seq]
    else:
        a_sequences += [seq]
print('Finish generating questions and answers.')

# pad_sequences
# q_maxlen = max(len(seq) for seq in q_sequences)
q_test_data = pad_sequences(q_sequences, maxlen=q_maxlen)
# a_maxlen = max(len(seq) for seq in a_sequences)
a_test_data = pad_sequences(a_sequences, maxlen=a_maxlen)
print('Finish padding sequences.')

# Load Chinese word vector
word_vec = {}
with open(word_vec_path, 'r', encoding='utf8') as f:
    ignore_first_line = True
    for line in f:
        if ignore_first_line:
            ignore_first_line = False
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_vec[word] = coefs
print('Found {} word vectors.'.format(len(word_vec)))

# Prepare Embedding Matrix
num_words = len(word_index)
embedding_matrix = np.zeros((num_words, EMBD_DIM))
for word,i in word_index.items():
    if i >= len(word_index):
        continue
    embedding_vector = word_vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print('Finish generating embedding matrix.')

# Prepare testing data
q_test = q_test_data
a_test = a_test_data

# TODO: test if work

# load model & predict
model = load_model(model_path)
output = model.predict([q_test, a_test])

# get the answer from predicted result
output = output.reshape(output.size // 6, 6)
ans = np.argmax(output, axis=1)

# print ans out
with open(output_path, 'w') as f:
    print('id,ans', file=f)
    for i, a in enumerate(ans):
        print('{},{}'.format(i+1, a), file=f)
