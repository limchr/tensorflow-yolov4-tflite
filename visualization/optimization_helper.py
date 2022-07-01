import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
print(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from visualization.helper import get_model
from absl import app, flags
from feature_vis.helper import *
import tensorflow as tf
import numpy as np
import time
import cv2
from feature_vis.config import *
import importlib.util
#output_neurons[5][0][8][6][0][6]


import datetime


tf.compat.v1.enable_eager_execution()








current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

optimize_targets = None
optimize_multiplier = None


# Define loss function
def calc_loss(img, model, start_image, tv, l1, l2, c, targets):
    global optimize_targets
    global optimize_multiplier
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)


    # if len(layer_activations) == 1:
    #  layer_activations = [layer_activations]
    # layer_activations = [layer_activations]
    losses_l1 = []
    losses_l2 = []
    for act,ot,om in zip(layer_activations,optimize_targets, optimize_multiplier):
        # diff = act - tf.math.multiply(tf.convert_to_tensor(target,dtype=tf.float32),tf.constant(10000,dtype=tf.float32))
        diff = act - ot
        diffm = diff * om

        # pltval = np.sort(act.numpy().flatten())
        # plt.plot(pltval)
        # plt.show()

        # diff_flat = tf.reshape(diff,[-1])
        # for ind in np.where(target.flat > 0)[0]:
        #     diff_flat[ind] *= 10000

        losses_l1.append(tf.math.reduce_mean(tf.abs(diffm)))
        losses_l2.append(tf.math.reduce_mean(tf.square(diffm)))
    loss_l1 = tf.reduce_sum(losses_l1)
    loss_l2 = tf.reduce_sum(losses_l2)


    # Regularization for color hist
    hist = color_histogram(img, 32)
    prev_hist = color_histogram(start_image, 32)
    hist_diff = tf.reduce_sum(tf.math.abs(hist - prev_hist))

    l1l = l1 * loss_l1
    l2l = l2 * loss_l2
    tvl = tv * tf.image.total_variation(img)
    hil = c * hist_diff

    return l1l + l2l + tvl + hil, l1l, l2l, tvl, hil


# DeepDream module
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    input_signature=(tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
                     tf.TensorSpec(shape=[], dtype=tf.int32),
                     tf.TensorSpec(shape=[], dtype=tf.float32),
                     tf.TensorSpec(shape=[], dtype=tf.float32),
                     tf.TensorSpec(shape=[], dtype=tf.float32),
                     tf.TensorSpec(shape=[], dtype=tf.float32),
                     tf.TensorSpec(shape=[], dtype=tf.int32),
                     tf.TensorSpec(shape=[], dtype=tf.float32),
                     tf.TensorSpec(shape=None, dtype=tf.float32)),
    def __call__(self, img, steps, step_size, tv, l1, l2, pad, c, targets):
        print("Tracing")
        loss = tf.constant(0.0,dtype=tf.float32)
        start_image = tf.identity(img)


        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss, l1l, l2l, tvl, hil = calc_loss(img, model=self.model, start_image=start_image, tv=tv, l1=l1, l2=l2, c=c, targets= targets)


            gradients = tape.gradient(loss, img)

            gradients /= tf.math.reduce_std(gradients) + 1e-8

            img = img - gradients * step_size
            img = crop_and_pad(img, pad)
            # img = tf.clip_by_value(img, -1, 1)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', tf.cast(loss, tf.float64), step=tf.cast(n, tf.int64))
                tf.summary.scalar('l1l', tf.cast(l1l, tf.float64), step=tf.cast(n, tf.int64))
                tf.summary.scalar('l2l', tf.cast(l2l, tf.float64), step=tf.cast(n, tf.int64))
                tf.summary.scalar('tvl', tf.cast(tvl, tf.float64), step=tf.cast(n, tf.int64))
                tf.summary.scalar('hil', tf.cast(hil, tf.float64), step=tf.cast(n, tf.int64))
                if n % 10 == 0:
                    tf.summary.image('dream', tf.expand_dims(tf.cast(img,tf.float64), axis=0), step=tf.cast(n, tf.int64))

        return loss, img


from visualization.image_helper import read_img_yolo

def get_starting_point(type,val=None):
    if type == "scalar":
        val = 0.5 if val is None else val
        img = np.full((416, 416, 3), val, dtype="float32")
    elif type == "random":
        img = np.random.rand(416, 416, 3).astype("float32")
    elif type == "image":
        img = read_img_yolo(val).astype("float32")
    else:
        print("unknown type "+type)
        exit(1)
    return img

# Normalize an image
def deprocess_dream(img):
    img = tf.clip_by_value(img, -1, 1)
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)




def main(argv):
    model = get_model()
    model_output = model.output


    path_small = model_output[1][0]
    path_small_target = np.zeros(path_small.shape,dtype=np.float)
    path_middle = model_output[3][0]
    path_middle_target = np.zeros(path_middle.shape,dtype=np.float)
    path_large = model_output[5][0]
    path_large_target = np.zeros(path_large.shape,dtype=np.float)




    optimize_neurons = [
        path_small,
        path_middle,
        path_large
    ]

    global optimize_targets
    global optimize_multiplier

    optimize_targets = [
        path_small_target,
        path_middle_target,
        path_large_target
    ]

    optimize_multiplier = []
    for ot in optimize_targets:
        tgc = np.zeros(ot.shape, dtype=np.float)
        # tgc[tgc>0] = 1000000
        # tgc[tgc==0] = 1
        # tgc[:,:,:,:4] = 1.0/500 # normalization of xywh (max activation of appr. 450)
        tgc[:,:,:,4] = 1
        optimize_multiplier.append(tgc)


    bbx = 6
    bby = 6
    anchorid = 1

    for i in [bbx]:
        for j in [bby]:
            optimize_targets[2][i][j][anchorid][0] = 100 # x
            optimize_targets[2][i][j][anchorid][1] = 100 # y
            optimize_targets[2][i][j][anchorid][2] = 100 # w
            optimize_targets[2][i][j][anchorid][3] = 100 # h
            optimize_targets[2][i][j][anchorid][4] = 1 # c
            optimize_targets[2][i][j][anchorid][5] = 1 # p
            optimize_multiplier[2][i][j][anchorid][:] = 1
            optimize_multiplier[2][i][j][anchorid][:4] = 1./500
            optimize_multiplier[2][i][j][anchorid][4] = 100


    img = get_starting_point('scalar', 0.5)


    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=model.input, outputs=optimize_neurons)

    # Create the DeepDream
    deepdream = DeepDream(dream_model)

    loss, img = deepdream(tf.convert_to_tensor(img, dtype=tf.float32),
                          tf.constant(50000, dtype=tf.int32),
                          tf.constant(0.05, dtype=tf.float32),
                          tv=tf.constant(0.000000025, dtype=tf.float32),
                          l1=tf.constant(0.2, dtype=tf.float32),
                          l2=tf.constant(0.0002, dtype=tf.float32),
                          pad=tf.constant(1, dtype=tf.int32),
                          c=tf.constant(0, dtype=tf.float32),
                          targets=tf.convert_to_tensor(img, dtype=tf.float32))

    plt.imshow(img)
    plt.show()

if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass
