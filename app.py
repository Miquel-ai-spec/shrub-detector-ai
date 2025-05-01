from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained model (example path)
model = tf.keras.models.load_model('shrub_model.h5')

# Define the class names (update accordingly)
class_names = ['Bayabas', 'Hagunoy', 'Otot-otot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data.get('image')
    
    # Decode the image
    header, encoded = image_data.split(',', 1)
    image_bytes = io.BytesIO(base64.b64decode(encoded))
    img = Image.open(image_bytes)
    
    # Preprocess the image for prediction (resize, normalize, etc.)
    img = img.resize((224, 224))  # Example resize
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    
    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
