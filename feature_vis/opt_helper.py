import numpy as np
from absl import flags
from absl.flags import FLAGS
import re, os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense
from absl import app, flags
from core.yolov4 import YOLO, decode_train
from core.config import cfg
from core import utils
from feature_vis.config import get_DIR_PATH, get_MODEL_PLUS_PATH
import sys
sys.path.insert(0, get_DIR_PATH())

from visualization import helper
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

def create_model_plus_one(model,save=True, path=get_MODEL_PLUS_PATH(), summary=True):
    model = insert_opt_layer(model)
    if save: model.save(path)
    if summary: model.summary()
    return model

def insert_opt_layer(model):
    config = model.get_config()

    for i in range(len(config["layers"])):
        if "trainable" in config["layers"][i]["config"].keys():
            config["layers"][i]["config"]["trainable"] = False

    layer_config = {'name': 'opt_layer',
                    'class_name': 'Conv2D',
                    'config':  {'name': 'opt_layer',
                                'trainable': True,
                                'dtype': 'float32',
                                'filters': 3,
                                'kernel_size': (416, 416),
                                'strides': (1, 1),
                                'padding': 'same',
                                'data_format': 'channels_last',
                                'dilation_rate': (1, 1),
                                'activation': 'relu',
                                'use_bias': False,
                                'kernel_initializer':  {'class_name': 'Ones', 'config':{}},
                                    #{'class_name': 'RandomNormal',
                                     #                   'config': {'mean': 0.0,
                                      #                             'stddev': 0.01,
                                       #                            'seed': None,
                                        #                           'dtype': 'float32'}},
                                'bias_initializer':  {'class_name': 'Constant',
                                                      'config': {'value': 0.0,
                                                                 'dtype': 'float32'}},
                                'kernel_regularizer':  {'class_name': 'L1L2',
                                                        'config': {'l1': 0.0,
                                                                   'l2': 0.0005000000237487257}},
                                'bias_regularizer': None,
                                'activity_regularizer': None,
                                'kernel_constraint': None,
                                'bias_constraint': None},
                    'inbound_nodes': [[['input_1', 0, 0, {}]]]}
    config["layers"].insert(1, layer_config)

    config["layers"][2]["inbound_nodes"] = [[['opt_layer', 0, 0, {}]]]

    new_model = tf.keras.Model().from_config(config, custom_objects={})
    new_model.layers[0].set_weights(model.layers[0].get_weights())
    for i in range(1, len(model.layers)):
        try:
            new_model.layers[i + 1].set_weights(model.layers[i].get_weights())
        except:
            print("ERROR while transferring weights!")
            print(f"model layer: {new_model.layers[i + 1].get_config()}")
            print(f"source_model: {model.layers[i].get_config()}")
            print()

    return new_model

def opt_layer_factory():
    return Conv2D(3, 1, activation='relu', input_shape=(416, 416, 3), name="opt_layer")

def insert_layer_nonseq(model, layer_regex, insert_layer_factory, insert_layer_name=None, position='after', regex=True):
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

        # Insert layer if name matches the regular expression or if names match
        regex_or_name = re.match(layer_regex, layer.name) if regex else (layer_regex == layer.name)
        if regex_or_name:
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                x = layer_input
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            ''' It did not work to set the name, also no work around could be found. 
            Hence, the name is set in the function already and this part is unneccessary
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)
            '''
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:

            '''print(type(layer))
            print(type(layer_input))
            print(layer_input)'''
            try:
                if bool(re.match(r"tf.concat*", layer.name)):
                    x = layer(layer_input, -1)
                elif bool(re.match(r"tf.__operators__.getitem*", layer.name)):
                    x = layer(layer_input)
                elif layer.name == "tf.nn.max_pool2d":
                    x = layer(layer_input, ksize=13, padding="SAME", strides=1)
                elif layer.name == "tf.nn.max_pool2d_1":
                    x = layer(layer_input, ksize=9, padding="SAME", strides=1)
                elif layer.name == "tf.nn.max_pool2d_2":
                    x = layer(layer_input, ksize=5, padding="SAME", strides=1)
                elif layer.name == "tf.image.resize":
                    x = layer(layer_input, size=(26, 26), method="bilinear")
                elif layer.name == "tf.image.resize_1":
                    x = layer(layer_input, size=(52, 52), method="bilinear")
                elif isinstance(layer_input, list):
                    x = layer(*layer_input)
                else:
                    x = layer(layer_input)
            except:
                print(layer.get_config())
                print(type(layer_input))
                print(layer_input)
                print(type(layer))
                print(layer.name)
                raise ValueError("Something went wrong")

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)
