from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer



from ModelNextWordPrediction import TransformerBlock, predict_next_phrase, predict_next_words

# Tải dữ liệu tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

total_words = len(tokenizer.word_index) + 1

# Tải mô hình
try:
    modelTF = load_model('modelTF.h5', custom_objects={'TransformerBlock': TransformerBlock})
    print("Model TF loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

try:
    modelLSTM = load_model('modelLSTM.h5')
    print("Model LSTM loaded successfully.")
except Exception as e:
    print(f"Error loading model LSTM: {str(e)}")
try:
    modelGRU = load_model('modelGRU.h5')
    print("Model GRU loaded successfully.")
except Exception as e:
    print(f"Error loading model GRU: {str(e)}")


# Khởi tạo ứng dụng Flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text')
        use_next_words = data.get('useNextWords', True)  # Lấy trạng thái từ yêu cầu
        max_sequence_len = 536

        if text:
            if use_next_words:
                # Dự đoán từ tiếp theo
                prediction_tf = predict_next_words(modelTF, tokenizer, text, max_sequence_len)
                prediction_lstm = predict_next_words(modelLSTM, tokenizer, text, max_sequence_len)
                prediction_gru = predict_next_words(modelGRU, tokenizer, text, max_sequence_len)
            else:
                # Dự đoán câu tiếp theo
                prediction_tf = predict_next_phrase(modelTF, tokenizer, text, max_sequence_len)
                prediction_lstm = predict_next_phrase(modelLSTM, tokenizer, text, max_sequence_len)
                prediction_gru = predict_next_phrase(modelGRU, tokenizer, text, max_sequence_len)
            # Dự đoán từ tiếp theo cho các mô hình khác
            

            return jsonify({
                'prediction_tf': prediction_tf,
                'prediction_lstm': prediction_lstm,
                'prediction_gru': prediction_gru,
                'prediction_tf_phrases': prediction_tf  # Trả về dự đoán câu
            })
        else:
            return jsonify({'error': 'No text provided'}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)