import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


with open('models/knn_model.pkl', 'rb') as f:
    knn, label_dict = pickle.load(f)


def preprocess_image(image_path, size=(32, 32)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, size)
        img = img.reshape(1, -1)
    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files')
        predictions = []
        filenames = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                img = preprocess_image(file_path)
                prediction = knn.predict(img)
                predictions.append(label_dict[prediction[0]])
                filenames.append(filename)

        return render_template('index.html', filenames=filenames, predictions=predictions)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.context_processor
def utility_processor():
    return dict(zip=zip)


if __name__ == '__main__':
    app.run(debug=True)
