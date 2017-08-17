import os
import jieba
import numpy as np
import keras
import keras.backend as K
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, Embedding, LSTM, GRU, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Lambda
from keras.layers.merge import Add, Dot, Concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# parameter
data_path = './data/training_data/subtitle_no_TC/下課花路米'
word_vec_path = './outisde_data/wiki.zh.vector'
EMBD_DIM = 400

# load training data
def load_training_data(data_path):
    """ Load training data files from the directory `data_path`."""
    text_data = []
    for i,filename in enumerate(os.listdir(data_path)):
        path = os.path.join(data_path, filename)
        print(path)
        with open(path, 'r', encoding='utf8') as f:
            text_data += f.read().splitlines()[:-1]
        text_data += [""]
        if i > -1:
            break
    return text_data
text_data = load_training_data(data_path)

print('\nFind {} sentences.\n'.format(len(text_data)))
# print(text_data[:10])
# print(text_data[-10:])

# split words and add " " between blanks
def split_words(text_list):
    for i,_ in enumerate(text_list):
        text_list[i] = " ".join(jieba.cut(text_list[i]))
        # TODO: add user define dict
        # e.g.
        #   下課花路米
    return text_list
text_data = split_words(text_data)

# print(text_data[:10])
# print(text_data[-10:])


# generate q_text_data & a_text_data
# q_text_data: 1~3
# a_text_data: 4
q_text_data = []
a_text_data = []
for i,_ in enumerate(text_data):
    if i < 3 or not text_data[i-3] or not text_data[i-2] or not text_data[i-1] or not text_data[i]:
        continue
    q_text_data += [ text_data[i-3] + text_data[i-2] + text_data[i-1] ]
    a_text_data += [ text_data[i] ]


# generate tokenizer for all data (q_train + a_train)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
# with open(json_path, 'w') as fp:
#     json.dump(tokenizer.word_index, fp)
# tokenizer.word_index = word_index
word_index = tokenizer.word_index
print('\nFound {} unique tokens.\n'.format(len(word_index)))


# generate sequences
def gen_sequences(text_data):
    sequences = tokenizer.texts_to_sequences(text_data)

    # TODO: concate words together here?

    maxlen = max(len(seq) for seq in sequences)
    train_data = pad_sequences(sequences, maxlen=maxlen)
    return train_data, maxlen

q_train_data, q_maxlen = gen_sequences(q_text_data)
a_train_data, a_maxlen = gen_sequences(a_text_data)


# Load Chinese word vector
print('Indexing word vectors.')
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


# TODO: make problems




def generate_model():
    q_input = Input(shape=(q_train.shape[1],))
    q_vec = Embedding(num_words, EMBD_DIM, weights=[embedding_matrix], trainable=False)(q_input)
    # TODO: maybe bidirectional rnn would be better
    q_vec = GRU(200, activation='elu', dropout=0.3)(q_vec)
    q_vec = Dropout(0.7)(q_vec)
    q_vec = Dense(100, activation='elu')(q_vec)
    q_vec = Dropout(0.7)(q_vec)
    q_vec = Dense(50, activation='elu')(q_vec)
    # q_vec = BatchNormalization()(q_vec)

    a_input = Input(shape=(a_train.shape[1],))
    a_vec = Embedding(num_words, EMBD_DIM, weights=[embedding_matrix], trainable=False)(a_input)
    a_vec = GRU(200, activation='elu', dropout=0.3)(a_vec)
    a_vec = Dropout(0.7)(a_vec)
    a_vec = Dense(100, activation='elu')(a_vec)
    a_vec = Dropout(0.7)(a_vec)
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
model = generate_model()
