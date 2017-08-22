def use_device(device):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return
use_device('gpu')  # cpu / gpu

import jieba
import numpy as np
import os
import random

import keras
from keras import backend as K
from keras.layers import Input, Dropout, Dense, Embedding, LSTM, GRU, merge
from keras.layers.core import Reshape, Lambda
from keras.layers.merge import Add, Dot, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# parameter
data_path = './preprocess/corpus.txt'
word_vec_path = './outisde_data/wiki.zh.vector'
EMBD_DIM = 400

# load training data
with open(data_path, 'r', encoding='utf8') as f:
    text_data = f.read().splitlines()
print('Found {} sentences.'.format(len(text_data)))

# generate tokenizer for all data (q_train + a_train)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
# with open(json_path, 'w') as fp:
#     json.dump(tokenizer.word_index, fp)
# tokenizer.word_index = word_index
word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

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
fake_ans_id = copy_shuffle(list(range(len(text_data))))

# generate sequences
sequences = tokenizer.texts_to_sequences(text_data)

# generate another one wrong answer
ans = []
q_sequences = []
a_sequences = []
rand_ans = np.random.randint(2, size=len(sequences))
for i,_ in enumerate(text_data):
    if i < 1 or not sequences[i-1] or not sequences[i]:
        continue
    q_sequences += [ sequences[i-1] ]
    q_sequences += [ sequences[i-1] ]
    if rand_ans[i] == 0:
        ans += [1]
        a_sequences += [ sequences[i] ]
        ans += [0]
        a_sequences += [ sequences[fake_ans_id[i]] ]
    else:
        ans += [0]
        a_sequences += [ sequences[fake_ans_id[i]] ]
        ans += [1]
        a_sequences += [ sequences[i] ]
print('Finish generating questions and answers.')

# concate 3 together & generate other 5 wrong answer
# q_3_sequences = []
# a_3_sequences = []
# ans_3 = []
# rand_ans = np.random.randint(6, size=len(sequences))
# for i,_ in enumerate(text_data):
#     if i < 3 or not sequences[i-3] or not sequences[i-2] or not sequences[i-1] or not sequences[i]:
#         continue
#     fake_ans_cnt = 0
#     for j in range(6):
#         q_3_sequences += [ sequences[i-3] + sequences[i-2] + sequences[i-1] ]
#         if j == rand_ans[i]:
#             ans_3 += [1]
#             a_3_sequences += [ sequences[i] ]
#         else:
#             ans_3 += [0]
#             a_3_sequences += [ sequences[fake_ans_id[fake_ans_cnt][i]] ]
#             fake_ans_cnt += 1

# pad_sequences
q_maxlen = max(len(seq) for seq in q_sequences)
q_train_data = pad_sequences(q_sequences, maxlen=q_maxlen)
a_maxlen = max(len(seq) for seq in a_sequences)
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

# TODO: split validation set
q_train = q_train_data
a_train = a_train_data
ans = np.array(ans)
ans = ans.reshape(ans.size, 1)

def generate_model(q_shape, a_shape):
    q_input = Input(shape=(q_shape,))
    q_vec = Embedding(num_words, EMBD_DIM, weights=[embedding_matrix], trainable=False)(q_input)
    # TODO: maybe bidirectional rnn would be better
    q_vec = GRU(200, activation='elu', dropout=0.3)(q_vec)
    q_vec = Dropout(0.5)(q_vec)
    q_vec = Dense(100, activation='elu')(q_vec)
    q_vec = Dropout(0.5)(q_vec)
    q_vec = Dense(50, activation='elu')(q_vec)
    # q_vec = BatchNormalization()(q_vec)

    a_input = Input(shape=(a_shape,))
    a_vec = Embedding(num_words, EMBD_DIM, weights=[embedding_matrix], trainable=False)(a_input)
    a_vec = GRU(200, activation='elu', dropout=0.3)(a_vec)
    a_vec = Dropout(0.5)(a_vec)
    a_vec = Dense(100, activation='elu')(a_vec)
    a_vec = Dropout(0.5)(a_vec)
    a_vec = Dense(50, activation='elu')(a_vec)
    # a_vec = BatchNormalization()(a_vec)

    # XXX: dont use dnn!!
    # merge_vec = Concatenate()([q_vec, a_vec])
    # hidden = Dense(150, activation='elu')(merge_vec)
    # hidden = Dropout(0.2)(hidden)
    # hidden = Dense(100, activation='elu')(hidden)
    # hidden = Dropout(0.2)(hidden)
    # output = Dense(1, activation='softmax')(hidden) # maybe sigmoid will be better

    # TODO: use cosine similarity
    # see here: https://github.com/fchollet/keras/issues/2672#issuecomment-218188051
    cos_distance = merge([q_vec, a_vec], mode='cos', dot_axes=1)    # magic dot_axes works here!
    cos_distance = Reshape((1,))(cos_distance)
    cos_similarity = Lambda(lambda x: 1-x)(cos_distance)

    model = Model([q_input, a_input], [cos_similarity])
    model.summary()
    return model
model = generate_model(q_train.shape[1], a_train.shape[1])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([q_train, a_train], ans, epochs=10000, batch_size=10, validation_split=0.1)
