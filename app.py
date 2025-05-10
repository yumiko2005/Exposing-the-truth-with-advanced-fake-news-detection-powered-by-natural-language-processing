from flask import Flask, request, jsonify
import pickle

app = Flask(_name_)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    label = 'Fake' if prediction == 0 else 'Real'

    return jsonify({'prediction': label})
