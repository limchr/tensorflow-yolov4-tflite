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
from visualization.helper import get_model, get_class_names
from visualization.parameter import imgs, out_path_optimization
from visualization.image_helper import read_img_yolo, draw_bbox, save_image_wo_norm

import re
from keras.models import Model, Sequential
from keras.layers import Conv2D, InputLayer
from tfkerassurgeon.operations import delete_layer, insert_layer, delete_channels


def main(argv):
    # set up paths
    setup_clean_directory(out_path_optimization)

    model = get_model()

    # inp = InputLayer(input_shape=(416, 416, 3))
    # addon = Conv2D(3, 1, activation='relu', input_shape=(416, 416, 3))(inp.output)
    # addon.output = model.layers[1]

    new_model = insert_layer(model, model.layers[1], Conv2D(3, 1, activation='relu', input_shape=(416, 416, 3)))

    # insert_layer_nonseq(model, 'conv2d', insert_layer_factory, 'asdf', position="replace")

    class_names = get_class_names()

def insert_layer_factory():
    return Conv2D(3, 1, activation='relu', input_shape=(416, 416, 3))

import re
from keras.models import Model

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass