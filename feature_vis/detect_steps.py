import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from feature_vis.config import *
from PIL import Image
import cv2
import numpy as np
import sys

def preprocess_image(image_or_path=os.path.join(get_DIR_PATH(),"data","kite.jpg"), batch=True):
    INPUT_SIZE = 416

    if isinstance(image_or_path, str):
        original_image = cv2.imread(image_or_path)
    else:
        original_image = image_or_path
    path = sys.path
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (INPUT_SIZE, INPUT_SIZE))
    image_data = image_data / 255.
    if batch==True:
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        return images_data

    return image_data

def get_infer(model_path=get_MODEL_PLUS_PATH()):
    saved_model_loaded = tf.keras.models.load_model(model_path)
    infer = saved_model_loaded.signatures['serving_default']
    return infer

def decode_output(pred_bbox):
    if isinstance(pred_bbox, dict):
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
    else:
        boxes = pred_bbox[:, :, 0:4]
        pred_conf = pred_bbox[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return pred_bbox

def draw_bboxes(pred_bbox, image_or_path=os.path.join(get_DIR_PATH(),"data","kite.jpg"), output_path="result.png"):
    if isinstance(image_or_path, str):
        original_image = cv2.imread(image_or_path)
    else:
        original_image = image_or_path

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = utils.draw_bbox(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, image)

def wrapper(image_or_path=os.path.join(get_DIR_PATH(),"data","kite.jpg"), model_path=get_MODEL_PLUS_PATH(), output_path="result.png"):
    images_data = preprocess_image(image_or_path)
    batch_data = tf.constant(images_data)

    infer = get_infer(model_path)
    pred_bbox = infer(batch_data)

    pred_bbox = decode_output(pred_bbox)
    draw_bboxes(pred_bbox, image_or_path, output_path)

    return 0

def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
