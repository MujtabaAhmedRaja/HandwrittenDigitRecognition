from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
try:
    mlp_classifier = joblib.load('mlp_classifier.pkl')
    print("✅ Model loaded successfully.")
except:
    print("❌ Could not load model.")
    mlp_classifier = None

def preprocess_image(file):
    image = Image.open(file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))         # Resize to 28x28
    image_array = np.array(image).astype('float32') / 255.0
    return image_array.flatten().reshape(1, -1)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if mlp_classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        processed = preprocess_image(image_file)
        prediction = mlp_classifier.predict(processed)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
