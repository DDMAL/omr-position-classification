import keras
import cv2 as cv
import numpy as np
import h5py

from keras.models import load_model
from keras.backend import image_data_format

def process_neumes(image, coords, avg_neume_height, position_model_path, type_model_path):

    labels = []
    bounding_boxes = []

    [height, width, channels] = image.shape

    for c in coords:
        bounding_box = image[
            c[0]-2*avg_neume_height:c[0]+c[2]+2*avg_neume_height,
            c[1]:c[1]+c[3]]
        resize = cv.resize(bounding_box, (30,120), interpolation = cv.INTER_AREA)
        resize = resize / 255.0
        bounding_boxes.append(resize)

    bounding_boxes = [(bb / 1) for bb in bounding_boxes]
    bounding_boxes = np.asarray(bounding_boxes).reshape(len(bounding_boxes),120,30,3)

    position_model = load_model(position_model_path)
    pos_predictions = position_model.predict(bounding_boxes)

    type_predictions = []

    if type_model_path != '':
        type_model = load_model(type_model_path)
        type_predictions = type_model.predict(bounding_boxes)

    return pos_predictions, type_predictions
