#-----------------------------------------------------------------------------
# Program Name:         position_classification.py
# Program Description:  Rodan wrapper for position classification
#-----------------------------------------------------------------------------

import numpy as np
import cv2 as cv
import fileinput

from rodan.jobs.base import RodanTask
from . import training_interface

class PositionTraining(RodanTask):
    name = 'Position Training'
    author = 'Evan Savage'
    description = 'With original manuscript image and bounding box gamera xml files as inputs, individually label each neume component label to use for model training'
    enabled = True
    category = 'OMR'
    interactive = True

    settings = {
        'job_queue': 'Python3'
    }

    input_port_types = (
        {'name': 'Original Image', 'minimum': 1, 'maximum': 1, 'resource_types': lambda mime: mime.startswith('image/')},
        {'name': 'GameraXML File', 'minimum': 1, 'maximum': 1, 'resource_types': ['application/gamera+xml']},
        # {'name': 'Position Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+h5']},
    )

    output_port_types = (
        {'name': 'Trained Position Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+h5']},
    )

    def get_my_interface(self, inputs, settings):

        input_img_path = inputs['Original Image'][0]['resource_path']

        data = {
            'title': 'Yeet',
            'image': training_interface.media_file_path_to_public_url(input_img_path)
        }

        return ('position_training.html', data)

    def run_my_task(self, inputs, settings, outputs):
        if '@done' not in settings:
            return self.WAITING_FOR_INPUT()

        # input_position_model_path = inputs['Position Model'][0]['resource_path']
        input_xml_path = inputs['GameraXML File'][0]['resource_path']
        input_img_path = inputs['Original Image'][0]['resource_path']

        image = cv.imread(input_img_path, True)

        output_position_model_path = outputs['Trained Position Model'][0]['resource_path']



        return True
