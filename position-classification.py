#-----------------------------------------------------------------------------
# Program Name:         position_classification.py
# Program Description:  Rodan wrapper for position classification
#-----------------------------------------------------------------------------

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import statistics

from rodan.jobs.base import RodanTask


class PositionClassification(RodanTask):
    name = "Automatic Position Classification"
    author = "Evan Savage"
    description = "Given a pre-trained model, manuscript image, and gamera xml file, this task performs position classification of all neume components on a page"
    enabled = True
    category = "OMR"
    interactive = False

    settings = {}

    input_port_types = (
        {"name": "Image", "minimum": 1, "maximum": 1, "resource_types": lambda mime: mime.startswith("image/")},
        {"name": "GameraXML File", "minimum": 1, "maximum": 1, "resource_types": ["application/gamera+xml"]},
        {"name": "Position Model", "minimum": 1, "maximum": 1, "resource_types": ["keras/model+hdf5"]}
    )

    output_port_types = (
        {"name": "Generic XML File", "minimum": 1, "maximum": 1, "resource_types": ["application/xml"]}
    )

    def run_my_task(self, inputs, settings, outputs):

        position_model_path = inputs['Position Model'][0]['resource_path']
        xml_path = inputs['GameraXML File'][0]['resource_path']
        img_path = inputs['Image'][0]['resource_path']

        image = cv.imread(img_path, True)

        return 0
