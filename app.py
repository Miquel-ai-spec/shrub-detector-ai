<<<<<<< HEAD
import os
import base64
import io
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
model = load_model('shrub_model.h5')
class_names = ['Bayabas', 'Hagunoy', 'Otot-otot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.form['image']
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    return f"Predicted Shrub: {predicted_class}"

if __name__ == '__main__':
  app = Flask(__name__)
=======
import os
import base64
import io
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
model = load_model('shrub_model.h5')
class_names = ['Bayabas', 'Hagunoy', 'Otot-otot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.form['image']
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    return f"Predicted Shrub: {predicted_class}"

if __name__ == '__main__':
  app = Flask(__name__)
>>>>>>> bac5aaf4572a502b24061e89bc243f57f1405348
