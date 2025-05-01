from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your model (adjust if you stored it in a subfolder)
model = tf.keras.models.load_model('shrub_model.h5')

# Define class names
class_names = ['Bayabas', 'Hagunoy', 'Otot-otot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "No image received"}), 400

        # Decode base64 image
        header, encoded = image_data.split(",", 1)
        image_bytes = io.BytesIO(base64.b64decode(encoded))
        image = Image.open(image_bytes).convert("RGB")

        # Preprocess the image
        image = image.resize((224, 224))  # or the size your model expects
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
