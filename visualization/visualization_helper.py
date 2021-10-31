
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




def make_gradcam_heatmap(img_array, model, last_conv_layer_name, head_i, grid_x, grid_y, anchor_i, bb_i, normalize):

    # head_i = 5 # which head (1,3,5) for different conv sizes
    # grid_x = 3
    # grid_y = 6
    # bb_i = 4 # xywhcp
    #


    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_layer, preds = grad_model(img_array)

        if anchor_i == None:
            anchor_i = int(tf.argmax(preds[head_i][0,grid_y,grid_x,:,4]))


        class_channel = preds[head_i][0, grid_y, grid_x, anchor_i, bb_i]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_layer)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    conv_layer = conv_layer[0]


    ## original
    # heatmap = conv_layer @ pooled_grads[..., tf.newaxis]
    # heatmap = tf.squeeze(heatmap)

    ## ours
    heatmap = conv_layer * grads[0]
    heatmap = tf.reduce_mean(heatmap, axis=2)

    ## experimental
    # heatmap = grads[0]
    # heatmap = tf.reduce_mean(heatmap, axis=2)


    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    if normalize:
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    else:
        heatmap = tf.maximum(heatmap, 0) * 1
    # plt.imshow(heatmap)
    # plt.show()

    from PIL import Image
    reshaped = np.array(Image.fromarray(heatmap.numpy()).resize(img_array.shape[1:3], Image.NEAREST))

    return reshaped

