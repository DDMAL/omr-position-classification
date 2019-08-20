#-----------------------------------------------------------------------------
# Program Name:         position_classification.py
# Program Description:  Rodan wrapper for position classification
#-----------------------------------------------------------------------------

import numpy as np
import cv2 as cv
import fileinput

from rodan.jobs.base import RodanTask
from . import model_processing as processing
from . import xml_update


class PositionClassification(RodanTask):
    name = 'Automatic Position Classification'
    author = 'Evan Savage'
    description = 'Given a pre-trained model, manuscript image, and gamera xml file, this task performs position classification of all neume components on a page'
    enabled = True
    category = 'OMR'
    interactive = False

    settings = {
        'job_queue': 'Python3'
    }

    input_port_types = (
        {'name': 'Original Image', 'minimum': 1, 'maximum': 1, 'resource_types': lambda mime: mime.startswith('image/')},
        {'name': 'GameraXML File', 'minimum': 1, 'maximum': 1, 'resource_types': ['application/gamera+xml']},
        {'name': 'Position Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+h5']},
        # {'name': 'Staff Image', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgb+png']}
    )

    output_port_types = (
        {'name': 'Generic XML File', 'minimum': 1, 'maximum': 1, 'resource_types': ['application/xml']},
    )

    def run_my_task(self, inputs, settings, outputs):

        input_position_model_path = inputs['Position Model'][0]['resource_path']
        input_xml_path = inputs['GameraXML File'][0]['resource_path']
        input_img_path = inputs['Original Image'][0]['resource_path']

        image = cv.imread(input_img_path, True)

        output_xml_path = outputs['Generic XML File'][0]['resource_path']

        labels = ['l1', 'l2', 'l3', 'l4', 's1', 's2', 's3', 's4', 's5']

        glyph_coords, avg_glyph_height = xml_update.get_glyph_coords(input_xml_path)

        predictions = processing.process_neumes(
            image,
            glyph_coords,
            avg_glyph_height,
            input_position_model_path)

        xml_update.write_output_xml(input_xml_path, output_xml_path, predictions)

        return True
