import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.python.saved_model import tag_constants
from absl import app, flags
from core.yolov4 import YOLO, decode, filter_boxes
from feature_vis.config import *
import sys

sys.path.insert(0, get_DIR_PATH())
from feature_vis.detect_steps import preprocess_image
from visualization import helper
from feature_vis.opt_helper import create_model_plus_one
from visualization.helper import get_model
import os
from feature_vis.detect_steps import *

import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

BATCH_SIZE = 1
NUM_CLASS = 80
PATH_IMAGES = os.path.join(get_VIS_DIR_PATH(), "random_training_data")
MODEL_PLUS_PATH = get_MODEL_PLUS_PATH()
TRAINED_MODEL_PLUS_PATH = get_TRAINED_MODEL_PLUS_PATH()


def main(argv):
    # create new plus model if requested or load the model
    if get_BUILD_NEW():
        print("Create new plus model...")
        source_model = get_model()
        model = create_model_plus_one(source_model, path=MODEL_PLUS_PATH)
    else:
        print("Load model...")
        model = tf.keras.models.load_model(MODEL_PLUS_PATH)

    # set layers not trainable except first one
    print("Set trainabel variable...")
    for i, layer in enumerate(model.layers):
        if i == 1:
            layer.trainable = True
        else:
            layer.trainable = False

    # get images and preprocess
    print("Get training data...")
    image_path_list = os.listdir(PATH_IMAGES)
    image_path_list = [os.path.join(PATH_IMAGES, path) for path in image_path_list]
    data = [preprocess_image(path, batch=False) for path in image_path_list][0]
    data = np.asarray(data).astype(np.float32)
    print(data.shape)

    # compile with loss
    # model.compile(loss=custom_loss_function, optimizer="adam")

    # define true y
    y = np.zeros(shape=(200, 2))

    # train model
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    print("Train model...")
    epochs = 5
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (data_x, data_y) in enumerate(zip(data, y)):
            print(f"\rStep {step}", end="")
            print(data_x)
            data_x = tf.expand_dims(data_x, 0)
            print(data_x)
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                pred = model(data_x, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = custom_loss_function(data_y, pred)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_variables)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # save the trained model
    print("Save model...")
    model.save(TRAINED_MODEL_PLUS_PATH)


def custom_loss_function(y_true, y_pred):
    output = []

    loss = y_pred[2][6][6][2][3]-0.5
    return loss

    for path in y_pred:
        if path._rank() == 5:
            output.append(path)
    y_true = get_y_true()
    diff = []
    for true, pred in zip(y_true, output):
        diff.append(tf.math.reduce_mean(tf.subtract(true, pred)))
    return tf.cast(tf.math.reduce_mean(diff), dtype=tf.float64)


def get_y_true():
    first_path = np.zeros(shape=(1, 52, 52, 3, 85))
    first_path[:, :, :, :, 3] = 1
    second_path = np.zeros(shape=(1, 26, 26, 3, 85))
    second_path[:, :, :, :, 3] = 1
    third_path = np.zeros(shape=(1, 13, 13, 3, 85))
    third_path[:, :, :, :, 3] = 1
    return [first_path, second_path, third_path]

# Concentrate on one bounding box?
# use standard yolo without new layer as in send notebook
# Compare net out to custom loss -> single neuron not as currently with all neurons
# -> Anchor Box -< oP consist of 5 neurons -> only look at h part -< compare to some fixed scalar.
# Feed gray input -> train for small number of epochs -> possible result: outline of shapes
# implement -> coding session
# Contact Cristian over email!


def decode_train_to_decode(bbox_tensors):
    '''As the models decoding step from helper.get_model is different to the original used for the final model of
    Yolov4 we implemented this function to allow to transform the output to the original format of the original model.
    herefore we can also use the detect function or its steps to check whether the new layer is properly added,'''

    selection = [bbox_tensors["tf.concat_11"], bbox_tensors["tf.concat_13"], bbox_tensors["tf.concat_15"]]
    bbox_tensors = []
    prob_tensors = []
    for t in selection:
        pred_xywh = t[:, :, :, :, :4]
        pred_conf = t[:, :, :, :, 4:5]
        pred_prob = t[:, :, :, :, 5:]

        pred_prob = pred_conf * pred_prob
        pred_prob = tf.reshape(pred_prob, (BATCH_SIZE, -1, NUM_CLASS))
        pred_xywh = tf.reshape(pred_xywh, (BATCH_SIZE, -1, 4))

        bbox_tensors.append(pred_xywh)
        prob_tensors.append(pred_prob)

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=0.2, input_shape=tf.constant([416, 416]))
    pred = tf.concat([boxes, pred_conf], axis=-1)

    return pred


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
