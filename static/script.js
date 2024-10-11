const inputTextArea = document.getElementById('inputText');
const loadingMessage = document.getElementById('loadingMessage');
const nextWordButton = document.getElementById('nextWordButton');
const nextPhraseButton = document.getElementById('nextPhraseButton');

let useNextWords = true; // Trạng thái dự đoán hiện tại

nextWordButton.addEventListener('click', () => {
    useNextWords = true; // Sử dụng hàm dự đoán từ
    nextWordButton.classList.add('active'); // Làm nổi bật nút dự đoán từ
    nextPhraseButton.classList.remove('active'); // Bỏ nổi bật nút dự đoán câu
});

nextPhraseButton.addEventListener('click', () => {
    useNextWords = false; // Sử dụng hàm dự đoán câu
    nextPhraseButton.classList.add('active'); // Làm nổi bật nút dự đoán câu
    nextWordButton.classList.remove('active'); // Bỏ nổi bật nút dự đoán từ
});

// Lắng nghe sự kiện 'input' để dự đoán mỗi khi có thay đổi
inputTextArea.addEventListener('input', async function () {
    const text = inputTextArea.value;

    if (text.trim() === "") {
        // Nếu không có văn bản, xóa dự đoán
        document.querySelector('#predictionTF').innerHTML = "<strong>Transformer:</strong><br>No prediction available.";
        document.querySelector('#predictionLSTM').innerHTML = "<strong>LSTM:</strong><br>No prediction available.";
        document.querySelector('#predictionGRU').innerHTML = "<strong>GRU:</strong><br>No prediction available.";
        return;
    }

    // Hiển thị thông báo đang tải
    loadingMessage.style.display = 'block';

    const response = await fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({ text, useNextWords }), // Gửi trạng thái dự đoán
        headers: { 'Content-Type': 'application/json' }
    });

    // Ẩn thông báo đang tải
    loadingMessage.style.display = 'none';

    // Xử lý phản hồi
    if (response.ok) {
        const result = await response.json();

        // Xóa tất cả các dự đoán cũ
        document.querySelector('#predictionTF').innerHTML = "<strong>Transformer:</strong><br>";
        document.querySelector('#predictionLSTM').innerHTML = "<strong>LSTM:</strong><br>";
        document.querySelector('#predictionGRU').innerHTML = "<strong>GRU:</strong><br>";

        // Cập nhật các dự đoán cho Transformer
        if (useNextWords) {
            const predictions = result.prediction_tf;
            predictions.forEach(prediction => {
                const div = document.createElement('span');
                div.className = 'prediction-text';
                div.innerHTML = prediction; // Dự đoán
                document.querySelector('#predictionTF').appendChild(div); // Thêm vào container
            });
        } else {
            // Hiển thị dự đoán câu (nếu có)
            const predictions = result.prediction_tf_phrases;
            predictions.forEach(prediction => {
                const div = document.createElement('span');
                div.className = 'prediction-text';
                div.innerHTML = prediction; // Dự đoán
                document.querySelector('#predictionTF').appendChild(div); // Thêm vào container
            });
        }

        // Cập nhật các dự đoán cho LSTM
        result.prediction_lstm.forEach(prediction => {
            const div = document.createElement('span');
            div.className = 'prediction-text';
            div.innerHTML = prediction;
            document.querySelector('#predictionLSTM').appendChild(div);
        });

        // Cập nhật các dự đoán cho GRU
        result.prediction_gru.forEach(prediction => {
            const div = document.createElement('span');
            div.className = 'prediction-text';
            div.innerHTML = prediction;
            document.querySelector('#predictionGRU').appendChild(div);
        });

    } else {
        const error = await response.json();
        document.getElementById('result').innerHTML = "<strong>Error:</strong> " + (error.error || "An error occurred.");
    }
});

// Để ngăn chặn hành động mặc định của form khi người dùng nhấn Enter
document.getElementById('predictionForm').onsubmit = function(event) {
    event.preventDefault();
};
