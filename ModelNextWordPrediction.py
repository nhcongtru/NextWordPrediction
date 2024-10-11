import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Định nghĩa TransformerBlock
class TransformerBlock(tf.keras.Model):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

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
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            embed_dim=config.get('embed_dim', 64),
            num_heads=config.get('num_heads', 4),
            ff_dim=config.get('ff_dim', 128),
            rate=config.get('rate', 0.1),
            **config
        )

# Hàm dự đoán
def predict_next_words(model, tokenizer, text, max_sequence_len):
    # Chuyển đổi văn bản thành các chỉ số
    sequence = tokenizer.texts_to_sequences([text])[0]

    # In ra chuỗi gốc và chiều dài
    print("Original sequence:", sequence)
    print("Length of original sequence:", len(sequence))

    # Kiểm tra chiều dài chuỗi và thực hiện padding
    if len(sequence) > max_sequence_len:
        sequence = sequence[-max_sequence_len:]

    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    
    # In ra hình dạng sau khi padding
    print("Padded sequence shape:", sequence.shape)
    
    # Dự đoán với mô hình
    predicted = model.predict(sequence, verbose=0)
    top_3_indices = np.argsort(predicted[0])[::-1][:3]  # Lấy 3 chỉ số có xác suất cao nhất

    # Tạo danh sách từ dự đoán
    output_words = []
    for index in top_3_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                output_words.append(word)
                break  # Thoát vòng lặp khi tìm thấy từ

    return output_words  # Trả về danh sách 3 từ dự đoán


def predict_next_phrase(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) > max_sequence_len - 1:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)

    top_3_indices = np.argsort(predicted[0])[::-1][:3]

    top_3_words = []
    for index in top_3_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_3_words.append(word)
                break

    output_sentences = []
    
    for word in top_3_words:
        new_sentence = text + " " + word
        
        for _ in range(2):
            token_list = tokenizer.texts_to_sequences([new_sentence])[0]
            
            if len(token_list) > max_sequence_len - 1:
                token_list = token_list[-(max_sequence_len - 1):]

            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

            predicted = model.predict(token_list, verbose=0)

            predicted_index = np.argmax(predicted, axis=-1)[0]
            output_word = ""
            for word, idx in tokenizer.word_index.items():
                if idx == predicted_index:
                    output_word = word
                    break
            
            new_sentence += " " + output_word
        
        output_sentences.append(new_sentence)

    return output_sentences 

