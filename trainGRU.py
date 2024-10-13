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

modelGRU = Sequential()
modelGRU.add(Embedding(total_words, 128, input_length=max_sequence_len-1))
modelGRU.add(GRU(256))
modelGRU.add(BatchNormalization())
modelGRU.add(Dropout(0.2))
modelGRU.add(Dense(total_words, activation='softmax'))
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
modelGRU.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
modelGRU.fit(X, y, epochs=150, verbose=1, callbacks=[early_stopping])
modelGRU.save('modelGRUv4.h5')