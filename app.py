from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('shrub_model.h5')

# Define class names
class_names = ['Bayabas', 'Hagunoy', 'Otot-otot']

@app.route('/')
def index():
    return render_template('index.html')  # your camera HTML

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        data_url = data.get('image')

        # Check if valid image data
        if not data_url or ',' not in data_url:
            return jsonify({'error': 'Invalid image data'}), 400

        # Decode base64 image
        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))  # adjust based on model input size

        # Preprocess for model
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions)]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
