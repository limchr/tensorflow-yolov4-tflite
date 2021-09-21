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

from visualization.image_helper import draw_bbox
from visualization.helper import get_model


import matplotlib.pyplot as plt
import cv2

from visualization.visualization_helper import make_gradcam_heatmap




from visualization_helper import make_gradcam_heatmap
from image_helper import save_image_wo_norm
from common.helper import setup_clean_directory




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


def get_filtered_images(cls, grid_x, grid_y, break_at, scale):
    trainset = Dataset(FLAGS, is_training=True)
    class_names = trainset.classes

    cls_i = list(class_names.keys())[list(class_names.values()).index(cls)]


    # trainset = [trainset.__next__() for i in range(5)]

    selected = []

    for t in trainset:
        # sample[ 0 = images / 1 = annotations, 0 = small objs / 1 = mid objs / 2 = big objs, 0 = grid / 1 = list ]

        if t[1][scale][0][0, grid_x, grid_y, :, 5 + cls_i].any():
            selected.append(t)
            print('found '+str(len(selected)))
            if len(selected) >= break_at:
                break

    return selected



def main(_argv):

    scale = 2 # 0 small, 1 mid, 2 large

    if scale == 0:
        num_grid_cells = 52
    elif scale == 1:
        num_grid_cells = 26
    elif scale == 2:
        num_grid_cells = 13



    grid_x = 6
    grid_y = 6

    break_at = 50
    cls = 'person'
    trainset = Dataset(FLAGS, is_training=True)
    class_names = trainset.classes
    cls_i = list(class_names.keys())[list(class_names.values()).index(cls)]

    selected = get_filtered_images(cls,grid_x,grid_y, break_at, scale=scale)

    model = get_model()



    plt_dir = 'figs_gradcam_multi'
    img_dir = os.path.join(plt_dir, cls)
    file_type = '.png'

    setup_clean_directory(img_dir)

    plot_layers = []
    # plot_layers += [1,2,3] # last layers of backbone
    # plot_layers += [75,76,77] # last layers of backbone
    # plot_layers += [103,104,105,106,107,108,109] # last layers of yolov4
    # plot_layers = [105,]

    plot_layers += [4,5,6] # last layers of yolov4


    neurons = 'xywhcp'

    # save_dir = os.path.join(img_dir, 'grad_cam_' + neuron_label + '_' + str(i))
    # setup_clean_directory(save_dir)

    results = np.zeros((len(selected), len(plot_layers), len(neurons), 416, 416))

    for img_i in range(len(selected)):
        for neuron_i, neuron in enumerate(neurons):
            for layer_i, layer in enumerate(plot_layers):
                ni = neuron_i
                if neuron == 'p':
                    ni = 5 + cls_i
                grad_cam = make_gradcam_heatmap(selected[img_i][0], model, "conv2d_"+str(layer), (scale*2)+1, grid_x, grid_y, None, ni, False)
                results[img_i, layer_i, neuron_i] = grad_cam

    results_mean = results.mean(axis=0)
    results_mean_normalized = results_mean / results_mean.max()
    from visualization.image_helper import grid_image
    grid_img = grid_image(results_mean_normalized)

    save_image_wo_norm(grid_img)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass