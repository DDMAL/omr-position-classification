#-----------------------------------------------------------------------------
# Program Name:         position_classification.py
# Program Description:  Rodan wrapper for position classification
#-----------------------------------------------------------------------------

import numpy as np
import cv2 as cv
import fileinput
import os

from rodan.jobs.base import RodanTask
from . import label_interface
from . import xml_update
from . import model_training

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
        {'name': 'Original Image',
         'minimum': 1, 'maximum': 1,
         'resource_types': lambda mime: mime.startswith('image/')},
        {'name': 'GameraXML File',
         'minimum': 1, 'maximum': 1,
         'resource_types': ['application/gamera+xml']},
        # {'name': 'Position Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+h5']},
    )

    output_port_types = (
        {'name': 'Trained Position Model',
         'minimum': 1, 'maximum': 1,
         'resource_types': ['keras/model+hdf5']},
        {'name': 'Extended GameraXML File',
        'minimum': 1, 'maximum': 1,
        'resource_types': ['application/xml']},
    )

    def get_my_interface(self, inputs, settings):

        input_img_path = inputs['Original Image'][0]['resource_path']
        input_xml_path = inputs['GameraXML File'][0]['resource_path']

        glyph_coords, avg_glyph_height = xml_update.get_glyph_coords(input_xml_path)

        interface_labels = ['s1','l1','s2','l2','s3','l3','s4','l4','s5']

        count = 0
        positions = []

        if '@classify' in settings or '@save' in settings:
            positions = settings['@user_input'][0]
            positions = [p.encode('ascii') for p in positions]
            count = int(settings['@user_input'][1])
            print(positions)

        data = {
            'title': 'Position Labeler',
            'image': label_interface.media_file_path_to_public_url(input_img_path),
            'labels': interface_labels,
            'glyph_coords': glyph_coords,
            'agh': avg_glyph_height,
            'count': count,
            'positions': positions,
        }

        return ('position_labeler.html', data)

    def run_my_task(self, inputs, settings, outputs):
        if '@done' not in settings:
            return self.WAITING_FOR_INPUT()


        # input_position_model_path = inputs['Position Model'][0]['resource_path']
        input_xml_path = inputs['GameraXML File'][0]['resource_path']
        input_img_path = inputs['Original Image'][0]['resource_path']

        input_xml_url = inputs['GameraXML File'][0]['resource_url']

        print(inputs['GameraXML File'])

        output_xml_path = outputs['Extended GameraXML File'][0]['resource_path']
        output_model_path = outputs['Trained Position Model'][0]['resource_path']

        types = []

        positions = settings['@user_input']
        glyph_coords, avg_glyph_height = xml_update.get_glyph_coords(input_xml_path)

        position_model = model_training.train_model(
            input_img_path,
            glyph_coords,
            positions,
            avg_glyph_height,
            output_model_path + '.hdf5')

        os.rename(output_model_path + '.hdf5', output_model_path)
        xml_update.write_label_xml(input_xml_path, output_xml_path, positions, types)

        return True

    def validate_my_user_input(self, inputs, settings, user_input):
        if 'save' in user_input:
            return {'@save': True, '@user_input': user_input['user_input']}
        elif 'classify' in user_input:
            return {'@classify': True, '@user_input': user_input['user_input']}
        elif 'complete' in user_input:
            return { '@done': True, '@user_input': user_input['user_input'] }
