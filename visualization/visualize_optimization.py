from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import math

from image_helper import draw_bbox

import matplotlib.pyplot as plt
import cv2




flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './data/yolov4.weights', 'pretrained weights')
# flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

flags.DEFINE_string('image_path', './input_images', 'path to input image')
flags.DEFINE_string('output_path', './output_images', 'path to output image')

from visualization_helper import make_gradcam_heatmap

from image_helper import save_image_wo_norm
from common.helper import setup_clean_directory


def get_model():
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)

    return model


def visualize_gradcam(model, image_data, target, img_num, class_names,p_filter=None):
    image_data = image_data[[0], :, :, :]

    plt_dir = 'figs_gradcam'
    img_dir = os.path.join(plt_dir, str(img_num))
    file_type = '.png'

    setup_clean_directory(img_dir)
    from image_helper import draw_bbox

    cell_size = 416/13
    cx = p_filter[2][1] *cell_size
    cy = p_filter[2][0] *cell_size

    # img = draw_bbox(image_data[0],[[cx,cy,cell_size,cell_size]])
    img = image_data[0]

    save_image_wo_norm(img, img_dir, 'base_img'+file_type)

    cl_ind = list(class_names.keys())[list(class_names.values()).index(p_filter[0])]

    plot_layers = []
    plot_layers += [1,2,3] # last layers of backbone
    plot_layers += [75,76,77] # last layers of backbone
    plot_layers += [103,104,105,106,107,108,109] # last layers of yolov4




    neurons = 'xywhcp'

    for n in range(len(neurons)):
        neuron_label = neurons[n]

        if neuron_label == 'p':
            n = 5 + cl_ind

        for i in plot_layers:
            grad_cam = make_gradcam_heatmap(image_data, model, "conv2d_"+str(i), 5, p_filter[2][1], p_filter[2][0], None, n)
            save_image_wo_norm(grad_cam, img_dir, 'grad_cam_'+neuron_label+'_'+str(i)+file_type)




def visualize_gradcam_multi(model, image_data, target, img_num, class_names,p_filter=None):
    image_data = image_data[[0], :, :, :]

    plt_dir = 'figs_gradcam_multi'
    img_dir = os.path.join(plt_dir, str(img_num))
    file_type = '.png'

    setup_clean_directory(img_dir)
    from image_helper import draw_bbox

    cell_size = 416/13
    cx = p_filter[2][1] *cell_size
    cy = p_filter[2][0] *cell_size

    # img = draw_bbox(image_data[0],[[cx,cy,cell_size,cell_size]])
    img = image_data[0]

    save_image_wo_norm(img, img_dir, 'base_img'+file_type)

    cl_ind = list(class_names.keys())[list(class_names.values()).index(p_filter[0])]

    plot_layers = []
    plot_layers += [1,2,3] # last layers of backbone
    plot_layers += [75,76,77] # last layers of backbone
    # plot_layers += [103,104,105,106,107,108,109] # last layers of yolov4
    plot_layers = [105,]



    neurons = 'xywhcp'



    for n in range(len(neurons)):
        neuron_label = neurons[n]
        for i in plot_layers:
            save_dir = os.path.join(img_dir, 'grad_cam_' + neuron_label + '_' + str(i))
            setup_clean_directory(save_dir)
            for x in range(13):
                for y in range(13):
                    if neuron_label == 'p':
                        n = 5 + cl_ind
                    grad_cam = make_gradcam_heatmap(image_data, model, "conv2d_"+str(i), 5, x, y, None, n)
                    base_filename_append = ('%.5d' % (y,)) + '_' + ('%.5d' % (x,))
                    save_image_wo_norm(grad_cam, save_dir, base_filename_append+file_type)





def main(_argv):

    trainset = Dataset(FLAGS, is_training=False)
    class_names = trainset.classes

    model = get_model()


    trainset = [trainset.__next__() for i in range(5)]




    p_filter = [
        ['sandwich','spoon',[3,6]],
        ['laptop', 'cat',[5,7]],
        ['person', 'skateboard',[6,6]],
        ['cat', 'bed',[7,7]],
        ['pizza', 'dining table',[9,5]],
    ]


    for i in range(len(trainset)):
        image_data = trainset[i][0]
        target = trainset[i][1]
        pf = p_filter[i]
        visualize_gradcam_multi(model, image_data, target, i, class_names,p_filter=pf)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass