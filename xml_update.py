import xml.etree.ElementTree as ET

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
