from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import tensorflow as tf 
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = "models/Food_CNN_New_Model_Final_5.h5"

model = tf.keras.models.load_model(MODEL_PATH)



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(50, 50))

    x = image.img_to_array(img)
    x = x/255
    
    classes = np.array(['dumplings', 'hot_dog', 'pizza', 'samosa', 'steak'])

    preds = model.predict_classes(x.reshape(1,50,50,3))
    return classes[int(preds)]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        result = model_predict(file_path, model)

        return str(result)
    return None


if __name__ == '__main__':
    app.run(debug=True)

