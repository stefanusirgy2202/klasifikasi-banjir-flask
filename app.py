import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Set the allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the pre-trained model
model = tf.keras.models.load_model('transfer_learning(81%).h5')

# Mapping of class indices to class labels
class_names = ['+ 40 Cm', '0 Cm - 20 Cm', '20 Cm - 40 Cm']


def allowed_file(filename):
    # Check if the file extension is allowed
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image):
    # Resize the image to the required input shape of the model
    image = image.resize((256, 256))

    # Convert the image to an array of pixels
    image_array = np.array(image)

    # Normalize the pixel values to be in the range [0, 1]
    normalized_image_array = image_array / 255.0

    # Expand the dimensions to match the model's input shape
    preprocessed_image = np.expand_dims(normalized_image_array, axis=0)

    return preprocessed_image


def classify_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make the prediction
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    return predicted_class


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Check if the file is allowed
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type')

    # Save the uploaded file to a temporary folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.root_path, 'uploads', filename)
    file.save(file_path)

    # Classify the image
    predicted_class = classify_image(file_path)

    # Remove the temporary file
    os.remove(file_path)

    return render_template('index.html', predicted_class=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
