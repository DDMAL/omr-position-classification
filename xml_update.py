import xml.etree.ElementTree as ET
import numpy as np
import uuid
import lxml.etree as etree


def get_glyph_coords(xml_path):

    glyph_coords = []
    glyph_count = 0
    glyph_height_sum = 0

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for glyph in root.find('glyphs'):
        id = uuid.uuid1()
        uly = int(glyph.get('uly'))
        ulx = int(glyph.get('ulx'))
        nrows = int(glyph.get('nrows'))
        ncols = int(glyph.get('ncols'))
        glyph_count += 1
        glyph_height_sum += nrows
        glyph_coords.append([uly, ulx, nrows, ncols])

    avg_glyph_height = int(glyph_height_sum/glyph_count)

    return glyph_coords, avg_glyph_height

def write_classification_xml(input_xml_path, output_xml_path, positions, types,
    staff_bb, avg_glyph_height, height):

    pos_labels = ['l1', 'l2', 'l3', 'l4', 's1', 's2', 's3', 's4', 's5']
    type_labels = ['clef.c', 'clef.f', 'custos', 'neume.inclinatum',
    'neume.oblique2', 'neume.oblique3', 'neume.oblique4',
    'neume.podatus2', 'neume.podatus3', 'neume.podatus4',
    'neume.punctum', 'neume.virga']

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(input_xml_path, parser)
    root = tree.getroot()

    container = root.find('glyphs')

    inc = 0
    for glyph in container:
        uly = int(glyph.get('uly'))
        nrows = int(glyph.get('nrows'))
        g_y_span = [uly-avg_glyph_height, uly+nrows+avg_glyph_height]
        position = str(pos_labels[np.argmax(positions[inc])])
        t = str(type_labels[np.argmax(types[inc])])
        conf_pos = str(round(max(positions[inc]) * 100, 2))
        conf_type = str(round(max(types[inc]) * 100, 2))

        type = etree.SubElement(glyph, 'type')
        type.attrib['name'] = t
        type.attrib['confidence'] = conf_type
        pe_el = etree.SubElement(glyph, 'pitch-estimation')
        staff = etree.SubElement(glyph, 'staff')
        pos_el = etree.SubElement(pe_el, 'position')
        pitch_el = etree.SubElement(pe_el, 'pitch')
        pos_el.attrib['name'] =  position
        pos_el.attrib['confidence'] = conf_pos
        j = 1
        for bb in staff_bb:
            if (g_y_span[1] > round(bb[0] * height)) and \
                (round(bb[2] * height) > g_y_span[0]):
                staff.attrib['number'] = str(j)
            j = j + 1
        inc = inc + 1

    container[:] = sorted(container,
        key=lambda g:[int(g.find('staff').get('number')), int(g.get('ulx'))])

    tuning = []

    pos_ordered = ['s1', 'l1', 's2', 'l2', 's3', 'l3', 's4', 'l4', 's5']

    c_pitch = [
        ['G', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'A'],
        ['E', 'F', 'G', 'A', 'B', 'C', 'D', 'E', 'F'],
        ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C', 'D']
    ]
    f_pitch = [
        ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C', 'D'],
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A', 'B'],
        ['F', 'G', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    ]

    for glyph in container:

        if glyph.find('type').get('name') == 'clef.c':
            if glyph.find('pitch-estimation').find('position').get('name') == 's2':
                tuning = c_pitch[0]
            elif glyph.find('pitch-estimation').find('position').get('name') == 's3':
                tuning = c_pitch[1]
            elif glyph.find('pitch-estimation').find('position').get('name') == 's4':
                tuning = c_pitch[2]
        elif glyph.find('type').get('name') == 'clef.f':
            if glyph.find('pitch-estimation').find('position').get('name') == 's2':
                tuning = f_pitch[0]
            elif glyph.find('pitch-estimation').find('position').get('name') == 's3':
                tuning = f_pitch[1]
            elif glyph.find('pitch-estimation').find('position').get('name') == 's4':
                tuning = f_pitch[2]
        else:
            i = pos_ordered.index(glyph.find('pitch-estimation').find('position').get('name'))
            pitch = tuning[i]
            glyph.find('pitch-estimation').find('pitch').attrib['name'] = pitch


    for bb in staff_bb:
        print(round(bb[0] * height), round(bb[2] * height))

    with open(output_xml_path, 'w') as out_file:
        out_file.write(etree.tostring(tree, pretty_print=True, encoding='unicode'))

    return True


def write_label_xml(input_xml_path, output_xml_path, url, positions, types):

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(input_xml_path, parser)
    root = tree.getroot()

    for glyph in root.find('glyphs'):
        position = positions[inc]
        type = etree.SubElement(glyph, 'type')
        type.attrib['name'] = ''
        pe_el = etree.SubElement(glyph, 'pitch-estimation')
        pos_el = etree.SubElement(pe_el, 'position')
        pitch_el = etree.SubElement(pe_el, 'pitch')
        pos_el.attrib['name'] = position
        pos_el.attrib['confidence'] = 1.00
        pitch_el.attrib['name'] = ''

    with open(output_xml_path, 'w') as out_file:
        out_file.write(etree.tostring(tree, pretty_print=True, encoding='unicode'))

    return True
