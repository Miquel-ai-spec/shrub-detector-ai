from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
import io

app = Flask(__name__)

# Load your model once when the app starts
model = tf.keras.models.load_model("shrub_model.h5")

# Define your class labels (adjust if your model differs)
class_names = ['Bayabas', 'Hagunoy', 'Otot-otot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Expect base64 image string in "image"
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        data_url = data['image']
        header, encoded = data_url.split(',', 1)  # Split off base64 header
        image_data = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
