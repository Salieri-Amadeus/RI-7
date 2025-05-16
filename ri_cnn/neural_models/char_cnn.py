from __future__ import print_function

import os
import sys
import logging
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Dense, Input, GlobalMaxPooling1D, Dropout,
    Conv1D, MaxPooling1D, Embedding
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from custom_metrics import hamming_score  # your existing hamming_score

# ------------------------------------------------------------------------------
# 1) HYPERPARAMS
# ------------------------------------------------------------------------------
conv_units   = int(sys.argv[1])         # e.g. 128
dropout_rate = float(sys.argv[2])       # e.g. 0.5
dense_units  = int(sys.argv[3])         # e.g. 256
max_len      = int(sys.argv[4])         # now = max # of chars

out_dir = 'res/char_cnn'
os.makedirs(out_dir, exist_ok=True)
logfile = f"{conv_units}_{dropout_rate}_{dense_units}_{max_len}.log"
logging.basicConfig(
    filename=os.path.join(out_dir, logfile),
    level=logging.INFO,
    format="%(message)s"
)

# ------------------------------------------------------------------------------
# 2) CHAR-CNN SPECIFIC
# ------------------------------------------------------------------------------
MAX_CHAR_LEN  = max_len                  # pad/trunc to this many characters
MAX_NUM_CHARS = 100                      # top 100 most frequent chars
CHAR_EMB_DIM  = 50                       # learn 50-dim char embeddings

# ------------------------------------------------------------------------------
# 3) LOAD TSVs
# ------------------------------------------------------------------------------
train_file = '../data/msdialog/train.tsv'
valid_file = '../data/msdialog/valid.tsv'
test_file  = '../data/msdialog/test.tsv'

def load_data_and_labels(data_file):
    texts, labels = [], []
    labels_map = ['OQ','RQ','FQ','IR','PF','NF','O','PA','GG','FD','CQ','JK']
    with open(data_file) as f:
        for line in f:
            if not line.strip(): continue
            tag_str, utt, *_ = line.rstrip().split('\t')
            texts.append(utt)
            arr = [0]*len(labels_map)
            for t in tag_str.split('_'):
                arr[labels_map.index(t)] = 1
            labels.append(arr)
    return texts, np.array(labels)

x_train, y_train = load_data_and_labels(train_file)
x_val,   y_val   = load_data_and_labels(valid_file)
x_test,  y_test  = load_data_and_labels(test_file)

# ------------------------------------------------------------------------------
# 4) CHAR TOKENIZER & PAD
# ------------------------------------------------------------------------------
tokenizer = Tokenizer(
    num_words=MAX_NUM_CHARS,
    char_level=True,
    oov_token=None
)
tokenizer.fit_on_texts(x_train + x_val)

seqs = tokenizer.texts_to_sequences(x_train + x_val + x_test)
data = pad_sequences(
    seqs,
    maxlen=MAX_CHAR_LEN,
    padding='post',
    truncating='post'
)

# split back
n1, n2 = len(x_train), len(x_val)
x_train_pad = data[:n1]
x_val_pad   = data[n1:n1+n2]
x_test_pad  = data[n1+n2:]

vocab_size = min(MAX_NUM_CHARS, len(tokenizer.word_index)+1)

# ------------------------------------------------------------------------------
# 5) BUILD 6-LAYER CHAR-CNN
# ------------------------------------------------------------------------------
inp = Input(shape=(MAX_CHAR_LEN,), dtype='int32')
emb = Embedding(
    input_dim=vocab_size,
    output_dim=CHAR_EMB_DIM,
    embeddings_initializer='uniform',
    input_length=MAX_CHAR_LEN,
    trainable=True
)(inp)

x = emb
for i, k in enumerate([7,7,3,3,3,3]):
    x = Conv1D(
        filters=conv_units,
        kernel_size=k,
        activation='relu',
        padding='same'
    )(x)
    if i < 2:
        x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(dropout_rate)(x)

x = GlobalMaxPooling1D()(x)
x = Dense(dense_units, activation='relu')(x)
x = Dropout(dropout_rate)(x)
out = Dense(y_train.shape[1], activation='sigmoid')(x)

model = Model(inp, out)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)

# ------------------------------------------------------------------------------
# 6) TRAIN
# ------------------------------------------------------------------------------
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(
    x_train_pad, y_train,
    validation_data=(x_val_pad, y_val),
    batch_size=128,
    epochs=20,
    callbacks=[es],
    verbose=2
)

# ------------------------------------------------------------------------------
# 7) EVALUATE & LOG
# ------------------------------------------------------------------------------
# get raw probabilities
y_val_prob  = model.predict(x_val_pad)
y_test_prob = model.predict(x_test_pad)

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for th in thresholds:
    y_val_pred  = (y_val_prob  >= th).astype(int)
    y_test_pred = (y_test_prob >= th).astype(int)

    # validation metrics
    acc_val = hamming_score(y_val, y_val_pred)
    p_val   = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    r_val   = recall_score   (y_val, y_val_pred, average='macro', zero_division=0)
    f1_val  = f1_score       (y_val, y_val_pred, average='macro', zero_division=0)

    # test metrics
    acc_test = hamming_score(y_test, y_test_pred)
    p_test   = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    r_test   = recall_score   (y_test, y_test_pred, average='macro', zero_division=0)
    f1_test  = f1_score       (y_test, y_test_pred, average='macro', zero_division=0)

    # exactly 13 comma-separated fields:
    logging.info(
        f"{conv_units},{dropout_rate},{dense_units},{max_len},"
        f"{th:.1f},"
        f"{acc_val:.4f},{p_val:.4f},{r_val:.4f},{f1_val:.4f},"
        f"{acc_test:.4f},{p_test:.4f},{r_test:.4f},{f1_test:.4f}"
    )

# ------------------------------------------------------------------------------
# 8) FINAL PRINT & SAVE
# ------------------------------------------------------------------------------
loss, acc = model.evaluate(x_test_pad, y_test, verbose=0)
print(f"\nTest  Loss: {loss:.4f}")
print(f"Test  Acc:  {acc:.4f}")

model.save(os.path.join(out_dir,
    f"char_cnn_{conv_units}_{dropout_rate}_{dense_units}_{max_len}.h5"
))
print("✅ Model saved.")
