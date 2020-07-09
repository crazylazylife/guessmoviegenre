from flask import Flask, request, render_template, url_for
from flask_cors import CORS
#   from werkzeug import secure_filename
import numpy as np
import cv2
import sys
import os
from keras.models import model_from_json
import skimage.transform
import json
import scipy

# load saved model
model = model_from_json(open("./model/gpre.json", "r").read())
model.load_weights("./code/gpre.h5")

app = Flask(__name__)
CORS(app)
#


def preprocessImg(img, size):
    img = skimage.transform.resize(img, size)
    img = img.astype(np.float32)
    #img = (img/127.5)-1
    # print(img)
    return img


@app.route("/")
def indexPage():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file found", 400

    filestr = request.files['file'].read()
    #filestr.save(os.path.join(uploads_dir, secure_filename(filestr.filename)))
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    image = img.copy()
    img_size = (150, 100, 3)
    img = preprocessImg(img, img_size)
    img = np.array(img, 'float32')
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)

    predictions = np.array(predictions)
    # print(predictions)
    ids = predictions[0].argsort()[::-1]
    # print(ids)
    ids = ids[:5]
    print(ids)

    ljson = open("./code/label.json", "r")
    labels = json.load(ljson)
    outputs = []
    for idx in ids:
        outputs.append([labels['id2genre'][idx],  predictions[0][idx]])
    ljson.close()
    return render_template("output.html", outputs=outputs)


if __name__ == "__main__":
    app.run(host="localhost", port="5000", threaded=True, debug=True)
