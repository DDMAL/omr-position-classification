#-----------------------------------------------------------------------------
# Program Name:         position_classification.py
# Program Description:  Rodan wrapper for position classification
#-----------------------------------------------------------------------------

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
# import statistics
import fileinput

from rodan.jobs.base import RodanTask
from . import model_processing as processing


class PositionClassification(RodanTask):
    name = 'Automatic Position Classification'
    author = 'Evan Savage'
    description = 'Given a pre-trained model, manuscript image, and gamera xml file, this task performs position classification of all neume components on a page'
    enabled = True
    category = 'OMR'
    interactive = False

    settings = {}

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

        glyph_count = 0
        glyph_height_sum = 0
        glyph_coords = []

        labels = ['l1', 'l2', 'l3', 'l4', 's1', 's2', 's3', 's4', 's5']

        tree = ET.parse(input_xml_path)
        root = tree.getroot()

        for glyph in root.find('glyphs'):
            uly = int(glyph.get('uly'))
            ulx = int(glyph.get('ulx'))
            nrows = int(glyph.get('nrows'))
            ncols = int(glyph.get('ncols'))
            glyph_count += 1
            glyph_height_sum += nrows
            glyph_coords.append([uly, ulx, nrows, ncols])

        avg_glyph_height = int(glyph_height_sum/glyph_count)

        predictions = processing.process_neumes(
            image,
            glyph_coords,
            avg_glyph_height,
            input_position_model_path)
        # print(labels)
        with open(input_xml_path, 'r') as in_file:
            buf = in_file.readlines()

        inc = 0

        with open(output_xml_path, 'w') as out_file:
            for line in buf:
                if "</ids>" in line:
                    position = str(labels[np.argmax(predictions[inc])])
                    confidence = str(max(predictions[inc]) * 100)
                    line = line + \
                        '\t\t\t<type name=""/>\n' + \
                        '\t\t\t<pitch-estimation>\n' + \
                        '\t\t\t\t<position name="' + position + \
                        '" confidence="' + confidence + '"/>\n' + \
                        '\t\t\t\t<pitch name=""/>\n' + \
                        '\t\t\t</pitch-estimation>\n'
                    inc += 1
                out_file.write(line)

            # out_file.write(labels)

        return True
