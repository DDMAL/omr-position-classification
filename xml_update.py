import xml.etree.ElementTree as ET
import numpy as np


def get_glyph_coords(xml_path):

    glyph_coords = []
    glyph_count = 0
    glyph_height_sum = 0

    tree = ET.parse(xml_path)
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

    return glyph_coords, avg_glyph_height


def write_output_xml(input_xml_path, output_xml_path, predictions):

    labels = ['l1', 'l2', 'l3', 'l4', 's1', 's2', 's3', 's4', 's5']


    with open(input_xml_path, 'r') as in_file:
        buf = in_file.readlines()

    inc = 0

    with open(output_xml_path, 'w') as out_file:
        for line in buf:
            if "</ids>" in line:
                position = str(labels[np.argmax(predictions[inc])])
                confidence = str(round(max(predictions[inc]) * 100, 2))
                line = line + \
                    '\t\t\t<type name=""/>\n' + \
                    '\t\t\t<pitch-estimation>\n' + \
                    '\t\t\t\t<position name="' + position + \
                    '" confidence="' + confidence + '"/>\n' + \
                    '\t\t\t\t<pitch name=""/>\n' + \
                    '\t\t\t</pitch-estimation>\n'
                inc += 1
            out_file.write(line)
        out_file.close()

    return True
