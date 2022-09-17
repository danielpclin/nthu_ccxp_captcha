import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import models, mixed_precision

from skimage import transform


def load(filename):
    img_width = 80
    img_height = 30
    np_image = Image.open(filename).convert('RGB')
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (img_height, img_width, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def predict():
    checkpoint_path = f'checkpoints/25.hdf5'
    alphabet = list('0123456789')
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    model = models.load_model(checkpoint_path)
    model.summary()
    image = load('auth_img.png')
    pred = model.predict(image)

    result = ["" for _ in range(len(pred[0]))]
    pred = np.argmax(pred, axis=2)
    for digit in pred:
        for index, code in enumerate(digit):
            result[index] = result[index] + int_to_char[code]
    print(result)


def main():
    predict()


if __name__ == "__main__":
    main()
