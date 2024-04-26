from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

app = Flask(__name__)
model = load_model('website/defungi-bs-128-e-150.h5')
class_names = ["H1","H2","H3","H5","H6"]  

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})

    img_path = 'temp_image.jpg'
    image_file.save(img_path)

    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
