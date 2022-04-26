import numpy as np
from flask import Flask, request, jsonify, render_template
import sys
import os
app = Flask(__name__)
app.debug = True
import json
import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

model = models.densenet121(pretrained=True)
model.eval()

imagenet_class_mapping = json.load(open('imagenet_class_index.json'))


def create_app(config_filename):
    app = Flask(__name__)
    app.config.from_pyfile(config_filename)
    return app


create_app('config.py')
app.config['UPLOAD_FOLDER'] = r'.\static\uploads'


@app.route('/')
def home():
    return render_template('file_upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        #cat = predict_image_class(path)
        #return "Image Category is:" + str(cat)
        return show_predictions(path)
    return "bad"


@app.route('/prediction')
def show_predictions(path):
    cat = predict_image_class(path)
    return render_template('Prediction_page.html', path=path, category=cat)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def predict_image_class(image_path):
    # read the image in binary form
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    # transform the image
    transformed_image = transform_image(image_bytes=image_bytes)
    # use the model to predict the class
    outputs = model.forward(transformed_image)
    _, category = outputs.max(1)
    # return the value
    predicted_idx = str(category.item())
    return imagenet_class_mapping[predicted_idx][1]


if __name__ == "__main__":
    app.run(debug=True)
