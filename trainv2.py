import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, MultiHeadAttention, LayerNormalization, Dropout, GRU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling1D

with open('datasetViet.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('\n'):
  if line.strip():
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

with open('tokenizerv2.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

modelLSTM = Sequential()
modelLSTM.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
modelLSTM.add(LSTM(256))
modelLSTM.add(BatchNormalization())
modelLSTM.add(Dropout(0.3))
modelLSTM.add(Dense(total_words, activation='softmax'))
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
modelLSTM.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
modelLSTM.fit(X, y, epochs=100, verbose=1, callbacks=[early_stopping])
modelLSTM.save('modelLSTMv2.h5')

modelGRU = Sequential()
modelGRU.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
modelGRU.add(GRU(256))
modelGRU.add(BatchNormalization())
modelGRU.add(Dropout(0.3))
modelGRU.add(Dense(total_words, activation='softmax'))
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
modelGRU.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
modelGRU.fit(X, y, epochs=100, verbose=1, callbacks=[early_stopping])
modelGRU.save('modelGRUv2.h5')

class TransformerBlock(Model):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, trainable=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.trainable = trainable

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'trainable': self.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        return cls(
            embed_dim=config.get('embed_dim', 64),
            num_heads=config.get('num_heads', 4),
            ff_dim=config.get('ff_dim', 128),
            rate=config.get('rate', 0.1),
            trainable=config.get('trainable', True),
            **config
        )

embed_dim = 64
num_heads = 4
ff_dim = 128

def build_model(total_words, max_sequence_len, embed_dim, num_heads, ff_dim):
    inputs = tf.keras.Input(shape=(max_sequence_len-1,))
    embedding_layer = Embedding(total_words, embed_dim, input_length=max_sequence_len-1)(inputs)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(embedding_layer, training=True)

    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(total_words, activation="softmax")(x)

    modelTF = Model(inputs=inputs, outputs=outputs)
    modelTF.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelTF

modelTF = build_model(total_words, max_sequence_len, embed_dim, num_heads, ff_dim)
modelTF.summary()

modelTF.fit(X, y, epochs=500, verbose=1)
modelTF.save('modelTFv2.h5')
