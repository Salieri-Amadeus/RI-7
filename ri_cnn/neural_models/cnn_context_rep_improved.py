# coding: utf-8

from __future__ import print_function

import os, sys, logging, pickle
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Dense, Input, GlobalMaxPooling1D, Dropout,
    Conv1D, MaxPooling1D, Embedding, concatenate, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from custom_metrics import hamming_score, f1

# ------------------------------------------------------------------------------
# Parse hyperparameters
# ------------------------------------------------------------------------------
conv_units          = int(sys.argv[1])    # e.g. 1024
dropout_rate        = float(sys.argv[2])  # e.g. 0.6
dense_units         = int(sys.argv[3])    # e.g. 256
max_len             = int(sys.argv[4])    # e.g. 50
context_conv_units  = int(sys.argv[5])    # e.g. 128
context_dense_units = int(sys.argv[6])    # e.g. 128

# ------------------------------------------------------------------------------
# Setup output directory & logging
# ------------------------------------------------------------------------------
out_dir = 'res/cnn_context_rep_improved'
os.makedirs(out_dir, exist_ok=True)
logfile = f"{conv_units}_{dropout_rate}_{dense_units}_{max_len}_{context_conv_units}_{context_dense_units}.log"
logging.basicConfig(
    filename=os.path.join(out_dir, logfile),
    level=logging.INFO,
    format="%(message)s"
)

# ------------------------------------------------------------------------------
# Embedding and vocabulary settings
# ------------------------------------------------------------------------------
GLOVE_DIR       = '../embeddings/'
EMBEDDING_FILE  = 'msdialog_w2v.txt'
MAX_NUM_WORDS   = 20000
EMBEDDING_DIM   = 100
EMBED_INIT_GLOVE = True

# ------------------------------------------------------------------------------
# File paths
# ------------------------------------------------------------------------------
train_file      = '../data/msdialog/train.tsv'
valid_file      = '../data/msdialog/valid.tsv'
test_file       = '../data/msdialog/test.tsv'
train_feat_file = '../data/msdialog/train_features.tsv'
valid_feat_file = '../data/msdialog/valid_features.tsv'
test_feat_file  = '../data/msdialog/test_features.tsv'

# ------------------------------------------------------------------------------
# Load pre-trained word embeddings
# ------------------------------------------------------------------------------
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, EMBEDDING_FILE), encoding='utf-8') as f:
    for line in f:
        values = line.rstrip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# ------------------------------------------------------------------------------
# Load text and multi-labels (12-way)
# ------------------------------------------------------------------------------
labels_map = ['OQ','RQ','FQ','IR','PF','NF','O','PA','GG','FD','CQ','JK']
def load_data(file):
    texts, labels = [], []
    with open(file) as f:
        for line in f:
            if not line.strip(): continue
            tag, utt = line.strip().split('\t')[:2]
            texts.append(utt)
            arr = [0]*len(labels_map)
            for t in tag.split('_'):
                arr[labels_map.index(t)] = 1
            labels.append(arr)
    return texts, np.array(labels)

x_train, y_train = load_data(train_file)
x_val,   y_val   = load_data(valid_file)
x_test,  y_test  = load_data(test_file)

# ------------------------------------------------------------------------------
# Load absolute-position feature for context boundaries
# ------------------------------------------------------------------------------
def load_abs_pos(file):
    pos = []
    with open(file) as f:
        for line in f:
            if not line.strip(): continue
            feats = line.strip().split('\t')[1].split()
            pos.append(int(feats[10]))
    return np.array(pos)

train_pos = load_abs_pos(train_feat_file)
val_pos   = load_abs_pos(valid_feat_file)
test_pos  = load_abs_pos(test_feat_file)

# ------------------------------------------------------------------------------
# Tokenize at word-level and pad
# ------------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train + x_val)
sequences = tokenizer.texts_to_sequences(x_train + x_val + x_test)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=max_len)
n1, n2 = len(x_train), len(x_val)

x_train_seq = data[:n1]
x_val_seq   = data[n1:n1+n2]
x_test_seq  = data[n1+n2:]

# ------------------------------------------------------------------------------
# Generate (prev, curr, next) arrays
# ------------------------------------------------------------------------------
def gen_ctx_arrays(x_seq, abs_pos):
    n, L = x_seq.shape
    ctx = np.zeros((n, L*3), dtype=int)
    for i in range(n):
        prev = x_seq[i-1] if i>0 and abs_pos[i]!=1 else np.zeros(L, dtype=int)
        curr = x_seq[i]
        nxt  = x_seq[i+1] if i<n-1 and abs_pos[i+1]!=1 else np.zeros(L, dtype=int)
        ctx[i] = np.hstack((prev, curr, nxt))
    return ctx[:, :L], ctx[:, L:2*L], ctx[:, 2*L:]

X_pre_train, X_curr_train, X_post_train = gen_ctx_arrays(x_train_seq, train_pos)
X_pre_val,   X_curr_val,   X_post_val   = gen_ctx_arrays(x_val_seq,   val_pos)
X_pre_test,  X_curr_test,  X_post_test  = gen_ctx_arrays(x_test_seq,  test_pos)

# ------------------------------------------------------------------------------
# Build embedding layer
# ------------------------------------------------------------------------------
num_words = min(MAX_NUM_WORDS, len(word_index)+1)
if EMBED_INIT_GLOVE:
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for w,i in word_index.items():
        if i<MAX_NUM_WORDS and w in embeddings_index:
            embedding_matrix[i] = embeddings_index[w]
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=True,
        name='embed'
    )
else:
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        input_length=max_len,
        trainable=True,
        name='embed'
    )

# ------------------------------------------------------------------------------
# Convolutional block helper
# ------------------------------------------------------------------------------
def conv_block(x, filters, kernel, pool, drop, prefix):
    # first conv + strided “pool” by using strides=pool
    x = Conv1D(filters, kernel, strides=1, activation='relu', padding='same', name=f'{prefix}_conv1')(x)
    x = Conv1D(filters, kernel, strides=pool, activation='relu', padding='same', name=f'{prefix}_down1')(x)
    x = Dropout(drop, name=f'{prefix}_drop1')(x)

    x = Conv1D(filters, kernel, strides=1, activation='relu', padding='same', name=f'{prefix}_conv2')(x)
    x = Conv1D(filters, kernel, strides=pool, activation='relu', padding='same', name=f'{prefix}_down2')(x)
    x = Dropout(drop, name=f'{prefix}_drop2')(x)

    x = Conv1D(filters, kernel, activation='relu', padding='same', name=f'{prefix}_conv3')(x)
    # replace GlobalMaxPooling1D with a plain reduce_max
    x = Lambda(lambda t: tf.reduce_max(t, axis=1), name=f'{prefix}_gmp')(x)
    x = Dropout(drop, name=f'{prefix}_drop3')(x)
    return x

# ------------------------------------------------------------------------------
# Build the three‐stream context‐rep model
# ------------------------------------------------------------------------------
inp_pre  = Input((max_len,), name='inp_prev')
inp_curr = Input((max_len,), name='inp_curr')
inp_post = Input((max_len,), name='inp_next')

emb_pre  = embedding_layer(inp_pre)
emb_curr = embedding_layer(inp_curr)
emb_post = embedding_layer(inp_post)

c_pre  = conv_block(emb_pre,  context_conv_units, 3, 3, dropout_rate, 'pre')
c_curr = conv_block(emb_curr, conv_units,         3, 3, dropout_rate, 'curr')
c_post = conv_block(emb_post, context_conv_units, 3, 3, dropout_rate, 'post')

x = concatenate([c_pre, c_curr, c_post], name='concat_all')
x = Dense(dense_units, activation='relu', name='dense_final')(x)
x = Dropout(dropout_rate, name='drop_final')(x)
out = Dense(len(labels_map), activation='sigmoid', name='out')(x)

model = Model([inp_pre, inp_curr, inp_post], out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.summary()

# ------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(
    [X_pre_train, X_curr_train, X_post_train], y_train,
    validation_data=([X_pre_val, X_curr_val, X_post_val], y_val),
    batch_size=128, epochs=50, callbacks=[es], verbose=2
)

# ------------------------------------------------------------------------------
# Evaluate & log thresholds
# ------------------------------------------------------------------------------
y_val_prob  = model.predict([X_pre_val, X_curr_val, X_post_val])
y_test_prob = model.predict([X_pre_test, X_curr_test, X_post_test])
for th in [0.2,0.3,0.4,0.5,0.6,0.7]:
    y_val_pred  = (y_val_prob  >= th).astype(int)
    y_test_pred = (y_test_prob >= th).astype(int)
    # ensure at least one label per sample
    for arr,prob in [(y_val_pred,y_val_prob),(y_test_pred,y_test_prob)]:
        for i in range(arr.shape[0]):
            if not arr[i].any():
                idx = np.argmax(prob[i])
                arr[i,idx] = 1
    acc_val,p_val_,r_val_,f1_val_ = hamming_score(y_val, y_val_pred), *f1(y_val, y_val_pred)
    acc_test,p_test_,r_test_,f1_test_ = hamming_score(y_test, y_test_pred), *f1(y_test, y_test_pred)
    logging.info(
        f"{conv_units},{dropout_rate},{dense_units},{max_len},{th:.1f},"
        f"{acc_val:.4f},{p_val_:.4f},{r_val_:.4f},{f1_val_:.4f},"
        f"{acc_test:.4f},{p_test_:.4f},{r_test_:.4f},{f1_test_:.4f}"
    )
    with open(os.path.join(out_dir, f"{conv_units}_{dropout_rate}_{dense_units}_{max_len}_{context_conv_units}_{context_dense_units}_{th:.1f}.res"), 'wb') as fp:
        pickle.dump(y_test_pred, fp)

# ------------------------------------------------------------------------------
# Save model
# ------------------------------------------------------------------------------
model.save(os.path.join(out_dir, f"cnn_context_rep_imp_{conv_units}_{dropout_rate}_{dense_units}_{max_len}.h5"))
print("✅ Improved CNN-Context-Rep model saved to", out_dir)
