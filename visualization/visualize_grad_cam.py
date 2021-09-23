from absl import app, flags, logging
from absl.flags import FLAGS
import os
from core.dataset import Dataset
import numpy as np

from visualization.helper import get_model


from visualization.visualization_helper import make_gradcam_heatmap

from visualization.visualization_helper import make_gradcam_heatmap
from visualization.image_helper import save_image_wo_norm, draw_bbox
from common.helper import setup_clean_directory



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

def grid_iterate(grid_x, grid_y, scale, scale_grid_cells, break_at, cls, plot_neurons, plot_layers, cls_i, model, plt_dir):
    selected = get_filtered_images(cls, grid_x, grid_y, break_at, scale=scale)

    neurons = 'xywhcp'

    results = np.zeros((len(selected), len(plot_layers), len(plot_neurons), 416, 416))

    for img_i in range(len(selected)):
        for neuron_i, neuron in enumerate(neurons):
            for layer_i, layer in enumerate(plot_layers):
                if neuron in plot_neurons:
                    ni = neuron_i
                    if neuron == 'p':
                        ni = 5 + cls_i
                    grad_cam = make_gradcam_heatmap(selected[img_i][0], model, "conv2d_" + str(layer), (scale * 2) + 1,
                                                    grid_x, grid_y, None, ni, False)
                    results[img_i, layer_i, plot_neurons.index(neuron)] = grad_cam

    results_mean = results.mean(axis=0)
    # results_mean_normalized = results_mean / results_mean.max()
    # from visualization.image_helper import grid_image
    # grid_img = grid_image(results_mean_normalized)

    img_base_name = 'img_%.5d_%.5d.jpg' % (grid_x,grid_y)
    for li,l in enumerate(plot_layers):
        for ni,n in enumerate(plot_neurons):
            img = results_mean[li,ni]
            img /= img.max()
            path = os.path.join(plt_dir,str(scale_grid_cells[scale]),str(l),n)
            save_image_wo_norm(img,path,img_base_name)


def main(_argv):

    # grid scales
    plot_scales = [2] # 0 small, 1 mid, 2 large
    scale_grid_cells = {0:52,1:26,2:13}
    scale_margins = {0:7,1:5,2:2}

    break_at = 10 # find break_at number of classes at each grid position
    cls = 'person' # the class to search for in the data set

    # which neurons should be visualized
    plot_neurons = 'wh'

    # read data set
    trainset = Dataset(FLAGS, is_training=True)
    class_names = trainset.classes
    cls_i = list(class_names.keys())[list(class_names.values()).index(cls)] # indexes of classes from dataset

    model = get_model()

    print('enter loop')

    plot_layers = []
    # plot_layers += [1,2,3] # last layers of backbone
    # plot_layers += [75,76,77] # last layers of backbone
    # plot_layers += [103,104,105,106,107,108,109] # last layers of yolov4
    # plot_layers = [105,]
    plot_layers += [75, 105]


    plt_dir = 'gradcam_out'
    for scale in plot_scales:
        for p in plot_layers:
            for n in plot_neurons:
                setup_clean_directory(os.path.join(plt_dir,str(scale_grid_cells[scale]),str(p),n))


    for scale in plot_scales:
        margin = scale_margins[scale]
        for x in range(margin,scale_grid_cells[scale]-margin-1):
            for y in range(margin,scale_grid_cells[scale]-margin-1):
                grid_iterate(x,y,scale, scale_grid_cells,break_at, cls, plot_neurons, plot_layers, cls_i, model, plt_dir)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass