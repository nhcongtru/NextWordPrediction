import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, MultiHeadAttention, LayerNormalization, Dropout, GRU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling1D
from underthesea import word_tokenize

with open('datasetViet4.txt', 'r', encoding='utf-8') as file:
    text = file.read()

def preprocess_text(text):
    text = text.lower()  # Chuyển sang chữ thường
    # Giữ lại dấu chấm (.), dấu phẩy (,) và dấu chấm hỏi (?) để phân đoạn câu
    text = re.sub(r'[^\w\s.,?]', '', text)  # Loại bỏ các ký tự đặc biệt, nhưng giữ lại dấu chấm, phẩy, hỏi
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    return text

# Bước 2: Tách từ tiếng Việt sử dụng Underthesea, xử lý từng câu
def vietnamese_tokenize(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Phân tách câu dựa trên dấu chấm và hỏi
    tokenized_sentences = [' '.join(word_tokenize(sentence)) for sentence in sentences if sentence.strip()]
    return '\n'.join(tokenized_sentences)

# Áp dụng xử lý trước văn bản
text = preprocess_text(text)
text = vietnamese_tokenize(text)

# Bước 3: Tokenizer và tạo các chuỗi n-gram
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")  # Thêm token đặc biệt cho từ ngoài từ điển
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('\n'):
    if line.strip():
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

# Bước 4: Padding các chuỗi n-gram
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Bước 5: Tách X và y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Bước 6: One-hot encoding cho y
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(f"Tổng số từ: {total_words}")
print(f"Kích thước X: {X.shape}")
print(f"Kích thước y: {y.shape}")
print(max_sequence_len)

with open('tokenizerv4.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

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

embed_dim = 64
num_heads = 8
ff_dim = 256

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

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

modelTF.fit(X, y, epochs=300, verbose=1, batch_size=64, callbacks=[early_stopping, lr_schedule])
modelTF.save('modelTFv4.h5')
