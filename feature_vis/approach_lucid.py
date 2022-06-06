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

CLI = flags.FLAGS
flags.DEFINE_spaceseplist(
    "neurons",  # name of the parameter
    "output_neurons[5][0][6][6][0][5]",
    "String of Neurons to Optimize, seperated by spaces, default is output_neurons[5][0][6][6][0][5]",
    short_name="n"
)
flags.DEFINE_spaceseplist(
    "img",  # name of the parameter
    "None",  # Gray img
    "String of path to single input image to use via os.path, seperated by spaces, default is a gray image",
    short_name="i"
)
flags.DEFINE_integer(
    "steps",  # name of the parameter
    1500,
    "Expects integer, Number of steps for neuron optimization, default 1500",
    short_name="s"
)
flags.DEFINE_float(
    "step_size",  # name of the parameter
    0.05,
    "Expects float as step size, default 0.01",
    short_name="sz"
)
flags.DEFINE_integer(
    "save_every",  # name of the parameter
    100,
    "Expects int, steps after which an intermediate image is saved, default 100",
    short_name="se"
)
flags.DEFINE_string(
    "file_name",  # name of the parameter
    "deep_dream_test",
    "Expects string, file name for saving images, default deep_dream_test",
    short_name="fn"
)
flags.DEFINE_spaceseplist(
    "file_path",  # name of the parameter
    "",
    "Expects list of strings seperated by spaces, file directory for saving images, default output_images",
    short_name="fp"
)
# Regularizations
flags.DEFINE_float(
    "total_variance",  # name of the parameter
    -0.000000025,  # Good balance between results and waiting time
    "Expects float, Total variance regularization, default=-0.000000025",
    short_name="tv"
)
flags.DEFINE_float(
    "lasso_1",  # name of the parameter
    0.2,  # Good balance between results and waiting time
    "Expects float, l1 regularization, default 0.2",
    short_name="l1"
)
flags.DEFINE_float(
    "lasso_2",  # name of the parameter
    2.0,  # Good balance between results and waiting time
    "Expects float, l2 regularization, default 2.0",
    short_name="l2"
)
flags.DEFINE_integer(
    "padding",  # name of the parameter
    1,  # Good balance between results and waiting time
    "Expects int, pad regularization, default 1",
    short_name="p"
)
flags.DEFINE_integer(
    "color_difference",  # name of the parameter
    0,  # Good balance between results and waiting time
    "Expects float, color hist regularization, pairs well with input img, default 0",
    short_name="c"
)
flags.DEFINE_bool(
    "reproduce",
    False,
    "Produces reproducable code with seed 2022, default false"
)


# Define loss function
def calc_loss(img, model, start_image, tv, l1, l2, c):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    # if len(layer_activations) == 1:
    #  layer_activations = [layer_activations]
    layer_activations = [layer_activations]
    losses_l1 = []
    losses_l2 = []
    for act in layer_activations:
        losses_l1.append(tf.math.reduce_mean(act))
        losses_l2.append(tf.math.reduce_mean(tf.square(act)))
    loss_l1 = tf.reduce_sum(losses_l1)
    loss_l2 = tf.reduce_sum(losses_l2)


    # Regularization for color hist
    hist = color_histogram(img, 32)
    prev_hist = color_histogram(start_image, 32)
    hist_diff = tf.reduce_sum(tf.math.abs(hist - prev_hist))



    # Regularization
    return (l1 * loss_l1) + (tv * tf.image.total_variation(img)) + (l2 * loss_l2) + (c * hist_diff)


# DeepDream module
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                         tf.TensorSpec(shape=[], dtype=tf.int32),
                         tf.TensorSpec(shape=[], dtype=tf.float32),
                         tf.TensorSpec(shape=[], dtype=tf.float32),
                         tf.TensorSpec(shape=[], dtype=tf.float32),
                         tf.TensorSpec(shape=[], dtype=tf.float32),
                         tf.TensorSpec(shape=[], dtype=tf.int32),
                         tf.TensorSpec(shape=[], dtype=tf.float32))
    )
    def __call__(self, img, steps, step_size, tv, l1, l2, pad, c):
        print("Tracing")
        loss = tf.constant(0.0)
        start_image = tf.identity(img)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, model=self.model, start_image=start_image, tv=tv, l1=l1, l2=l2, c=c)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)

            img = img + gradients * step_size
            img = crop_and_pad(img, pad)
            # img = tf.clip_by_value(img, -1, 1)

        return loss, img


# Main Loop
def run_deep_dream_simple(img, deepdream, steps=100, step_size=0.01, save_every=50,file_path = os.path.join("feature_vis","output_images"), file_name = str(int(time.time())), tv=0., l1=0., l2=0., pad=0., c=0,
                          na="",  show_img = True):
    # Convert from uint8 to the range expected by the model.
    img = tf.convert_to_tensor(img)
    param_anno = f"tv={tv}, l1={l1}, l2={l2}, pad={pad}"
    step_size = tf.convert_to_tensor(step_size, dtype="float32")
    tv = tf.convert_to_tensor(tv, dtype="float32")
    l1 = tf.convert_to_tensor(l1, dtype="float32")
    l2 = tf.convert_to_tensor(l2, dtype="float32")
    pad = tf.convert_to_tensor(pad, dtype="int32")
    c = tf.convert_to_tensor(c, dtype="float32")

    steps_remaining = steps
    step = 0
    if show_img: show(img, step, anno=f"{na}, {param_anno}")
    while steps_remaining:
        if steps_remaining > save_every:
            run_steps = tf.constant(save_every)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size), tv=tf.constant(tv), l1=tf.constant(l1),
                              l2=tf.constant(l2), pad=tf.constant(pad), c=tf.constant(c))
        if show_img: show(deprocess(img), step=step, anno=f"{na}, {param_anno}")
        save(img=deprocess(img), path=os.path.join(file_path, file_name + f"_{step}"), step=step,
             anno=f"{na}, {param_anno}")
        print("Step {}, loss {}".format(step, loss), end="\r")
    result = deprocess(img)
    if show_img: show(result, step=steps, anno=f"{na}, {param_anno}")
    save(img=result,
         path=os.path.join(file_path,file_name + "_final"),
         step=steps,
         anno=f"{na}, {param_anno}")
    print("Saved and finished")

    return result


def deep_dream_with_anchors(neurons = ["output_neurons[5][0][6][6][0][5]"], img = ["None"], steps = 1500, step_size = 0.05, save_every = 100,file_name = "deep_dream_test", file_path = "",tv = -0.000000025,l1 = 0.2, l2 = 2.0,pad = 1, c = 0, reproduce = False, model = None):
    if reproduce:
        tf.random.set_seed(2022)
    if file_path is str:
        file_path = os.path.join(os.path.curdir, "feature_vis", "output_images", file_path)
    elif file_path is list:
        file_path = os.path.join(os.path.curdir,"feature_vis","output_images", *file_path)

    # Get the model
    if model is None:
        base_model = get_model()
    else:
        base_model = model

    if file_path == []:
        file_path = ""
    img, _ = get_starting_point(img)
    # Choose layers to maximize
    # names = ['conv2d_4']
    # layers = [base_model.get_layer(name).output for name in names]
    # or choose a neuron
    output_neurons = base_model.output
    layers = []
    for s in set(neurons):
        layers.append(eval(s))
    # Choose annotation for layer/neuron

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    # Create the DeepDream
    deepdream = DeepDream(dream_model)

    dream_img = run_deep_dream_simple(
        img=img,
        deepdream=deepdream,
        steps=steps,
        step_size=step_size,
        save_every=save_every,
        # Regularizations
        tv=tv,
        l1=l1,
        l2=l2,
        pad=pad,
        c=c,
        # Annotations:
        file_path=file_path,
        file_name=file_name)

def main(argv):
    if CLI.reproduce:
        tf.random.set_seed(2022)
    # Set hyperparameters
    STEPS = CLI.steps
    STEP_SIZE = CLI.step_size
    SAVE_EVERY = CLI.save_every
    FILE_PATH = os.path.join(os.path.curdir,"feature_vis","output_images", *CLI.file_path)
    if not os.path.exists(FILE_PATH):
        os.mkdir(FILE_PATH)
    FILE_NAME = CLI.file_name
    TV = CLI.total_variance
    L1 = CLI.lasso_1
    L2 = CLI.lasso_2
    PAD = CLI.padding
    C = CLI.color_difference
    if CLI.img[0] == "None":
        IMG, SP_ANNO = get_starting_point(type="grey")
    elif CLI.img[0] == "Random":
        IMG, SP_ANNO = get_starting_point(type="random")
    else:
        IMG, SP_ANNO = get_starting_point(os.path.join(*CLI.img))


    # Get the model
    base_model = get_model()

    # Choose layers to maximize
    # names = ['conv2d_4']
    # layers = [base_model.get_layer(name).output for name in names]

    # or choose a neuron
    output_neurons = base_model.output
    layers = []
    for s in CLI.neurons:
        layers.append(eval(s))

    # Choose annotation for layer/neuron
    NAMES_ANNO = f"{CLI.neurons} {SP_ANNO}"

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    # Create the DeepDream
    deepdream = DeepDream(dream_model)

    dream_img = run_deep_dream_simple(
        img=IMG,
        deepdream=deepdream,
        steps=STEPS,
        step_size=STEP_SIZE,
        save_every= SAVE_EVERY,
        #Regularizations
        tv=TV,
        l1=L1,
        l2=L2,
        pad=PAD,
        c=C,
        #Annotations:
        na = NAMES_ANNO,
        file_path= FILE_PATH,
        file_name = FILE_NAME)


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass
