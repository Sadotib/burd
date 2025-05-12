from flask import Flask, request, jsonify
from predict import predict_bird

app = Flask(__name__)

@app.route('/')
def index():
    return "Bird Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        prediction, confidence = predict_bird(file)  # pass file-like object
        print("this is prediction: ",prediction,"and this is conf: ",type(confidence))
        return jsonify({
        'prediction': prediction,
        'confidence': confidence
        })
    except Exception as e:
        print(f"Error occurred in /predict route: {e}")  # <-- Add this
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
