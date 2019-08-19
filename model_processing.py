import keras
import cv2 as cv
import numpy as np

from keras.models import load_model
from keras.backend import image_data_format

def process_neumes(image, coords, avg_neume_height, model_path):

    labels = []

    model = load_model(model_path)

    [height, width, channels] = image.shape

    for c in coords:
        bounding_box = image[
            c[0]-2*avg_neume_height:c[0]+c[2]+2*avg_neume_height,
            c[1]:c[1]+c[3]]
        resize = cv.resize(bounding_box, (30,120), interpolation = cv.INTER_AREA)
        prediction = model.predict(resize)
        label = np.argmax(prediction)
        labels.append(label)


    return labels