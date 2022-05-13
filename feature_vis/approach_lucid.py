import os.path
import argparse
from visualization.helper import get_model
from absl import app, flags, logging
from feature_vis.helper import *
import tensorflow as tf
import numpy as np
import time
import cv2
from feature_vis.config import *


###
###
# Example use
# python approach_lucid.py
# --neurons "output_neurons[5][0][6][6][0][5]" /// allows multiple seperated by space / default "output_neurons[5][0][6][6][0][5]"
# --img "inputs" "001.jpg" /// parse path in strings, results in "inputs/001.jpg" /default "None" (gray image)
#
#

###
###  updates_every=50, tv=0, l1=0, l2=0, pad=0


# thanks to https://stackoverflow.com/a/64259328
def range_limited_float(mini, maxi):
    """Return function handle of an argument type function for
           ArgumentParser checking a float range: mini <= arg <= maxi
             mini - minimum acceptable argument
             maxi - maximum acceptable argument"""

    def range_limited_float_checker(arg):
        """ Type function for argparse - a float within some predefined bounds """
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("Argument must be < " + str(mini) + "and > " + str(maxi))
        return f

    return range_limited_float_checker


CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--neurons",  # name of the parameter
    "-n",
    nargs="+",  # creates a list
    type=str,  # each is a string, so we need to call eval on it
    default="output_neurons[5][0][6][6][0][5]",  # optimize central smallest bounding box of the biggest head for human
    help="String of Neurons to Optimize, seperated by spaces, default is output_neurons[5][0][6][6][0][5] "
)
CLI.add_argument(
    "--img",  # name of the parameter
    "-i",
    nargs="*",  # creates a list
    type=str,  # each is a folder, so we need to use os.path.join later TODO:Implement this
    default="None",  # optimize central smallest bounding box of the biggest head for human
    help="String of path to single input image to use via os.path, seperated by spaces, default is a gray image "
)
CLI.add_argument(
    "--steps",  # name of the parameter
    "-s",
    nargs=1,  # single input
    type=int,  # type declaration
    default=1500,  # Good balance between results and waiting time
    help="Number of steps for neuron optimization, default 1500"
)
CLI.add_argument(
    "--step_size",  # name of the parameter
    "-sz",  # alternative input
    nargs=1,  # single input
    type=range_limited_float(0.00000001, 0.99999999),  # custom type above that only allow between 0.[0*7]1 -> 0.[9*8]
    default=0.05,  # Good balance between results and waiting time
    help="(0,1) float as step size, default 0.01"
)
CLI.add_argument(
    "--save_every",  # name of the parameter
    "-se",  # alternative input
    nargs=1,  # single input
    type=int,
    default=100,  # Good balance between results and waiting time
    help="[0-steps] int, steps after which an intermediate image is saved"
)
# Regularizations
CLI.add_argument(
    "--total_variance",  # name of the parameter
    "-tv",  # alternative input
    nargs=1,  # single input
    type=range_limited_float(0., 0.1),
    default=-0.000000025,  # Good balance between results and waiting time
    help="[0-0.11] float, Total variance regularization default=-0.000000025"
)
CLI.add_argument(
    "--lasso_1",  # name of the parameter
    "-l1",  # alternative input
    nargs=1,  # single input
    type=range_limited_float(0., 0.99),
    default=0.2,  # Good balance between results and waiting time
    help="(0,0.99) float, lasso 1 regularization"
)
CLI.add_argument(
    "--lasso_2",  # name of the parameter
    "-l2",  # alternative input
    nargs=1,  # single input
    type=int,
    choices=range(0,10),
    default=2,  # Good balance between results and waiting time
    help="(0,10) int, lasso 2 regularization"
)
CLI.add_argument(
    "--padding",  # name of the parameter
    "-p",  # alternative input
    nargs=1,  # single input
    type=int,
    choices=range(0,10),
    default=1,  # Good balance between results and waiting time
    help="(0,10) int, pad regularization"
)


# Define loss function
def calc_loss(img, model, tv, l1, l2):
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

    # Regularization
    return (l1 * loss_l1) + (tv * tf.image.total_variation(img)) + (l2 * loss_l2)


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
                         tf.TensorSpec(shape=[], dtype=tf.int32))
    )
    def __call__(self, img, steps, step_size, tv, l1, l2, pad):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model, tv=tv, l1=l1, l2=l2)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = crop_and_pad(img,
                               pad)  # TODO: Observe whether it might be better to use crop and pad only every x steps
            # img = tf.clip_by_value(img, -1, 1)

        return loss, img


# Main Loop
def run_deep_dream_simple(img, deepdream ,steps=100, step_size=0.01, updates_every=50, tv=0, l1=0, l2=0, pad=0,
                          na=""):
    # Convert from uint8 to the range expected by the model.
    img = tf.convert_to_tensor(img)
    param_anno = f"tv={tv}, l1={l1}, l2={l2}, pad={pad}"
    step_size = tf.convert_to_tensor(step_size, dtype="float32")
    tv = tf.convert_to_tensor(tv, dtype="float32")
    l1 = tf.convert_to_tensor(l1, dtype="float32")
    l2 = tf.convert_to_tensor(l2, dtype="float32")
    pad = tf.convert_to_tensor(pad, dtype="int32")

    steps_remaining = steps
    step = 0
    show(img, step, anno=f"{na}, {param_anno}")
    while steps_remaining:
        if steps_remaining > updates_every:
            run_steps = tf.constant(updates_every)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size), tv=tf.constant(tv), l1=tf.constant(l1),
                              l2=tf.constant(l2), pad=tf.constant(pad))
        show(deprocess(img), step, anno=f"{na}, {param_anno}")
        print("Step {}, loss {}".format(step, loss), end="\r")
    result = deprocess(img)
    show(result, step=steps, anno=f"{na}, {param_anno}")
    save(img=result,
         path=os.path.join(os.path.curdir, "output_images", f"dream_img_{int(time.time())}.jpg"),
         step=steps,
         anno=f"{na}, {param_anno}")
    print("Saved and finished")

    return result
def main(argv):
    # argv structure:
    # 0 = file name, list of neurons
    args, unknown_args = CLI.parse_known_args()
    print(CLI.parse_args(["-n", "a", "b"]))
    print(args, " \n those are unknown:", unknown_args)

    # Set hyperparameters
    NEURONS = args.neurons
    STEPS = args.steps
    STEP_SIZE = args.step_size
    SAVE_EVERY = args.save_every
    TV = args.total_variance
    L1 = args.lasso_1
    L2 = args.lasso_2
    PAD = args.padding
    if type(args.img) == str:
        if args.img == "None":
            IMG, SP_ANNO = get_starting_point(type="grey")
        elif args.img == "Random":
            IMG, SP_ANNO = get_starting_point(type="random")
    else:
        IMG, SP_ANNO = get_starting_point(os.path.join(*args.img))
    PARAM_ANNO = f"tv={TV}, l1={L1}, l2={L2}, pad={PAD}"


    # Get the model
    base_model = get_model()

    # Choose layers to maximize
    # names = ['conv2d_4']
    # layers = [base_model.get_layer(name).output for name in names]

    # or choose a neuron
    output_neurons = base_model.output
    layers = [eval(s) for s in NEURONS] if type(NEURONS) == list else eval(NEURONS)

    # Choose annotation for layer/neuron
    NAMES_ANNO = f"{NEURONS} {SP_ANNO}"

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    # Create the DeepDream
    deepdream = DeepDream(dream_model)

    dream_img = run_deep_dream_simple(
        img=IMG,
        deepdream=deepdream,
        steps=STEPS,
        step_size=STEP_SIZE,
        updates_every= SAVE_EVERY, #TODO: Still shows images instead of saving them as a gif
        #Regularizations
        tv=TV,
        l1=L1,
        l2=L2,
        pad=PAD,
        #Annotations:
        na = NAMES_ANNO)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TODO: Remove before running on cluster!

    try:
        app.run(main)
    except SystemExit:
        pass
