import os
import sys
def use_device(device):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if device == 0 or device == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif device == 1 or device == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return
if len(sys.argv) < 2:
    use_device(0)  # 0 / 1 / 2 / ...
else:
    use_device(sys.argv[1])
    print('Using device {}'.format(sys.argv[1]))

import jieba
import json
import numpy as np
import random

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dropout, Dense, Embedding, LSTM, GRU, Bidirectional, merge
from keras.layers.core import Reshape, Lambda
from keras.layers.merge import Add, Dot, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# parameter
ID = 100

print("\nID = {}\n".format(ID))
model_path = './model/model_{}.h5'.format(ID)
weights_path = './weights/weights_{}.weights'.format(ID)
word_index_path = './word_index/{}.json'.format(ID)
SAVE_LOAD = 'save'

data_path = './preprocess/corpus.txt'
word_vec_path = './outside_data/wiki.zh.vector'
EMBD_DIM = 400
# word_vec_path = './outside_data/my.cbow.200d.txt'
# EMBD_DIM = 200
q_maxlen = 172
a_maxlen = 172
# a_maxlen = 72

# load training data
with open(data_path, 'r', encoding='utf8') as f:
    text_data = f.read().splitlines()
# text_data = text_data[ : len(text_data) // 50 ]
print('Found {} sentences.'.format(len(text_data)))

# generate tokenizer for all data (q_train + a_train)
tokenizer = Tokenizer()
if SAVE_LOAD == 'save': #save
    tokenizer.fit_on_texts(text_data)
    with open(word_index_path, 'w') as f:
        json.dump(tokenizer.word_index, f)
    word_index = tokenizer.word_index
elif SAVE_LOAD == 'load':   # load
    with open(word_index_path) as f:
        word_index = json.load(f)
    tokenizer.word_index = word_index
print('Found {} unique tokens.'.format(len(word_index)))

n_wrong_ans = 1

# generate wrong answers' index
def copy_shuffle(origin_list):
    shuffled_list = list(origin_list)
    random.shuffle(shuffled_list)
    # The for-loop below makes sure that there is only one correct answer for each question.
    for i, _ in enumerate(origin_list):
        if origin_list[i] == shuffled_list[i]:
            if i != len(origin_list) - 1:
                shuffled_list[i], shuffled_list[i + 1] = shuffled_list[i + 1], shuffled_list[i]
            else:
                shuffled_list[i], shuffled_list[0] = shuffled_list[0], shuffled_list[i]
    print('Finish Generating fake answers.')
    return shuffled_list
fake_ans_id = []
for _ in range(n_wrong_ans):
    fake_ans_id += [ copy_shuffle(list(range(len(text_data)))) ]

# generate sequences
sequences = tokenizer.texts_to_sequences(text_data)

# generate q & a sequences
q_sequences = []
a_sequences = []
ans = []

# concate 3 together & generate other N wrong answer
rand_ans = np.random.randint(n_wrong_ans + 1, size=len(sequences))
for i,_ in enumerate(sequences):
    if i < 3 or not sequences[i-3] or not sequences[i-2] or not sequences[i-1] or not sequences[i]:
        continue
    fake_ans_cnt = 0
    for j in range(n_wrong_ans + 1):
        # append a q_sequence
        q_sequences += [ sequences[i-3] + sequences[i-2] + sequences[i-1] ]
        # append an a_sequence
        if j == rand_ans[i]:
            ans += [1]
            a_sequences += [ sequences[i] ]
        else:
            ans += [0]
            a_sequences += [ sequences[fake_ans_id[fake_ans_cnt][i]] ]
            fake_ans_cnt += 1

print('Finish generating questions and answers.')

# pad_sequences
# q_maxlen = max(len(seq) for seq in q_sequences)
q_train_data = pad_sequences(q_sequences, maxlen=q_maxlen)
# a_maxlen = max(len(seq) for seq in a_sequences)
a_train_data = pad_sequences(a_sequences, maxlen=a_maxlen)
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
print('Preparing embedding matrix.')
num_words = len(word_index)
embedding_matrix = np.zeros((num_words, EMBD_DIM))
for word,i in word_index.items():
    if i >= len(word_index):
        continue
    embedding_vector = word_vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

q_train = q_train_data
a_train = a_train_data
ans = np.array(ans)
ans = ans.reshape(ans.size, 1)

input_dim = q_train.shape[1]

def create_base_model(input_dim):
    seq = Sequential()
    seq.add(Embedding(num_words, EMBD_DIM, weights=[embedding_matrix], trainable=False, input_shape=(input_dim, )))
    seq.add(Bidirectional(GRU(256)))
    return seq
base_model = create_base_model(input_dim)

input_q = Input(shape=(input_dim,))
input_a = Input(shape=(input_dim,))
processed_q = base_model(input_q)
processed_a = base_model(input_a)

# use cosine similarity
cos_distance = Dot(axes=1, normalize=True)([processed_q, processed_a])
cos_similarity = Reshape((1,))(cos_distance)
model = Model([input_q, input_a], [cos_similarity])
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=weights_path, save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max', verbose=1)
earlystopping = EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=1)
model.fit([q_train, a_train], ans, epochs=60, batch_size=1024, validation_split=0.1, callbacks=[checkpoint, earlystopping])

model.load_weights(weights_path)
model.save(model_path)