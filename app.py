import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np
from urllib.request import urlopen
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import time

# Seed random generators:
np.random.seed(42)
tf.random.set_seed(42)

# Load SavedModel from TensorFlow Hub:
model = hub.KerasLayer("./")


def get_image_from_url(url, read_flag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, read_flag)
    return image


def get_image_from_local_path(path, read_flag=cv2.IMREAD_COLOR):
    image = cv2.imread(path, read_flag)
    return image


INPUT_IMG_HEIGHT = 512
INPUT_IMG_WIDTH = 512
INPUT_CHANNEL_COUNT = 3


def preprocess_image(image):
    h, w, channel_count = image.shape
    if channel_count > INPUT_CHANNEL_COUNT:
        image = image[..., :INPUT_CHANNEL_COUNT]
    x = cv2.resize(
        image, (INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT), interpolation=cv2.INTER_AREA
    )
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x, h, w


def postprocess_mask(probability, h, w):
    probability = cv2.resize(probability, dsize=(w, h), interpolation=cv2.INTER_AREA)
    probability = np.expand_dims(probability, axis=-1)
    return probability


def create_masked_image(image, probability):
    alpha_image = np.dstack((image, np.full((image.shape[0], image.shape[1]), 255.0)))
    PROBABILITY_THRESHOLD = 0.7
    masked_image = np.where(probability > PROBABILITY_THRESHOLD, alpha_image, 0.0)
    return masked_image


app = Flask(__name__)


@app.route("/remove_bg", methods=["POST"])
def remove_bg():
    file = request.files["image"]
    filename = secure_filename(file.filename)
    tmp_dir = "/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    image_path = os.path.join(tmp_dir, filename)
    file.save(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    start_time = time.time()
    x, h, w = preprocess_image(image)
    probability = model(x)[0].numpy()
    probability = postprocess_mask(probability, h, w)
    masked_image = create_masked_image(image, probability)

    output_path = os.path.join(tmp_dir, "output.png")
    cv2.imwrite(output_path, masked_image)

    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time} seconds")

    with open(output_path, "rb") as f:
        img_data = f.read()

    return (img_data, 200, {"Content-Type": "image/png"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
