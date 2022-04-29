from absl import app, flags, logging
from absl.flags import FLAGS
import os
from core.dataset import Dataset
import numpy as np

from visualization.helper import get_model
import matplotlib.pyplot as plt

from visualization.visualization_helper import make_gradcam_heatmap

from visualization.visualization_helper import make_gradcam_heatmap
from visualization.image_helper import save_image_wo_norm, draw_bbox
from common.helper import setup_clean_directory
from visualization.helper import get_model, get_class_names
from visualization.parameter import imgs, out_path_optimization
from visualization.image_helper import read_img_yolo, draw_bbox, save_image_wo_norm

import re
from keras.models import Model, Sequential
from keras.layers import Conv2D, InputLayer
from tfkerassurgeon.operations import delete_layer, insert_layer, delete_channels

from tensorflow.keras import backend as K


import tensorflow

tensorflow.compat.v1.disable_eager_execution()



def main(argv):
    # set up paths
    # setup_clean_directory(out_path_optimization)

    model = get_model()

    def gradient_descent(x, iterations, step, max_win=None):
        for i in range(iterations):
            win_value, grad_values = fetch_win_and_grads([x])
            if max_win is not None and win_value > max_win:
                break
            if i % 5 == 0:
                print('Win at iteration', i, ':', win_value)
            x -= step * grad_values
        return x
    def gradient_ascent(x, iterations, step, max_win=None):
        for i in range(iterations):
            win_value, grad_values = fetch_win_and_grads([x])
            if max_win is not None and win_value > max_win:
                break
            if i % 5 == 0:
                print('Win at iteration', i, ':', win_value)
            x += step * grad_values
        return x

    ###
    # optimizing for a high confidence of single anchor box
    ###

    output_neurons = model.output
    win = K.square(output_neurons[5][0][6][6][0][4])

    input_neurons = model.input
    grads = K.gradients(win, input_neurons)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
    outputs = [win, grads]
    fetch_win_and_grads = K.function([input_neurons], outputs)


    starting_point = np.full((416, 416, 3), 0.5)  # 0.5 is medium gray
    plt.figure()
    plt.imshow(starting_point)


    feed_to_net = np.expand_dims(starting_point, axis=0)
    result_from_net = gradient_ascent(feed_to_net,
            iterations=500,
            step=0.005)


    ideal_img = np.clip(np.copy(result_from_net[0]), 0, 1.0)
    plt.figure()
    plt.imshow(ideal_img)
    plt.show()

    save_image_wo_norm(ideal_img,'out_opti','high_conf.jpg')

    ###
    # optimizing for a high probability of a big cat exist in the center of the image
    ###

    output_neurons = model.output
    win = K.square(output_neurons[5][0][6][6][0][0])

    input_neurons = model.input
    grads = K.gradients(win, input_neurons)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
    outputs = [win, grads]
    fetch_win_and_grads = K.function([input_neurons], outputs)


    starting_point = np.full((416, 416, 3), 0.5)  # 0.5 is medium gray
    plt.figure()
    plt.imshow(starting_point)


    feed_to_net = np.expand_dims(starting_point, axis=0)
    result_from_net = gradient_ascent(feed_to_net,
            iterations=500,
            step=0.005)


    ideal_img = np.clip(np.copy(result_from_net[0]), 0, 1.0)
    plt.figure()
    plt.imshow(ideal_img)
    plt.show()

    save_image_wo_norm(ideal_img,'out_opti','high_human.jpg')



    ###
    # optimize for a specific height of object
    ###
    step_size = 0.1
    for target_h in np.arange(0,1+step_size,step_size):

        # target_h = 0.6

        output_neurons = model.output
        win = K.square(output_neurons[5][0][6][6][0][3])

        input_neurons = model.input
        grads = K.gradients(K.square(win-target_h), input_neurons)[0]
        grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
        outputs = [win, grads]
        fetch_win_and_grads = K.function([input_neurons], outputs)




        starting_point = np.full((416, 416, 3), 0.5)  # 0.5 is medium gray
        plt.figure()
        plt.imshow(starting_point)


        feed_to_net = np.expand_dims(starting_point, axis=0)
        result_from_net = gradient_descent(feed_to_net,
                iterations=150,
                step=0.005)


        ideal_img = np.clip(np.copy(result_from_net[0]), 0, 1.0)
        plt.figure()
        plt.imshow(ideal_img)
        plt.show()

        save_image_wo_norm(ideal_img,'out_opti','height_%f.jpg'%(target_h,))


    ###
    # optimize for a specific x shift of small object
    ###
    step_size = 0.1
    for target_x in np.arange(0,1+step_size,step_size):

        # target_h = 0.6

        output_neurons = model.output
        win = K.square(output_neurons[1][0][6][6][0][0])

        input_neurons = model.input
        grads = K.gradients(K.square(win-target_x), input_neurons)[0]
        grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
        outputs = [win, grads]
        fetch_win_and_grads = K.function([input_neurons], outputs)




        starting_point = np.full((416, 416, 3), 0.5)  # 0.5 is medium gray
        plt.figure()
        plt.imshow(starting_point)


        feed_to_net = np.expand_dims(starting_point, axis=0)
        result_from_net = gradient_descent(feed_to_net,
                iterations=150,
                step=0.005)


        ideal_img = np.clip(np.copy(result_from_net[0]), 0, 1.0)
        plt.figure()
        plt.imshow(ideal_img)
        plt.show()

        save_image_wo_norm(ideal_img,'out_opti','x_%f.jpg'%(target_x,))



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass