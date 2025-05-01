from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import base64

app = Flask(__name__)

class ShrubClassifier:
    def __init__(self, model_path, labels):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = labels

    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))  # Adjust if your model needs a different size
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_bytes):
        processed = self.preprocess_image(image_bytes)
        prediction = self.model.predict(processed)
        return self.labels[np.argmax(prediction)]

labels = ['Hagunoy', 'Bayabas', 'Otot-otot']
classifier = ShrubClassifier('shrub_model.h5', labels)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_url = request.form['image']
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)

        predicted_label = classifier.predict(img_bytes)
        return render_template('result.html', prediction=predicted_label, image_data=data_url)
    except Exception as e:
        return f"Prediction failed: {e}"

if __name__ == '__main__':
    app.run(debug=True)
