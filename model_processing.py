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


def get_staff_bounding_boxes(staff_image_path, inference_graph_path):

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # detection_graph, sess = load_inference_graph(inference_graph_path)
    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # CWD_PATH = os.getcwd()
    PATH_TO_LABELS = os.path.dirname(__file__) + '/labelmap.pbtxt'
    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image = cv.imread(staff_image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image, axis=0)

    height, width, channels = image.shape

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    scores = np.squeeze(scores)
    staff_bb = []

    for i, box in enumerate(np.squeeze(boxes)):
        if box[0] != 0 and scores[i] > 0.95:
            staff_bb.append(np.insert(box,0,0))

    staff_bb_filter, x_start, x_end = filter_staff_bb(staff_bb)

    return staff_bb_filter
