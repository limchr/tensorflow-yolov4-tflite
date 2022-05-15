import os.path
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
import  importlib.util

CLI = flags.FLAGS
flags.DEFINE_spaceseplist(
    "neurons",  # name of the parameter
    "output_neurons[5][0][6][6][0][5]",
    "String of Neurons to Optimize, seperated by spaces, default is output_neurons[5][0][6][6][0][5]",
    short_name="n"
)
flags.DEFINE_spaceseplist(
    "img",  # name of the parameter
    "None",  # optimize central smallest bounding box of the biggest head for human
    "String of path to single input image to use via os.path, seperated by spaces, default is a gray image",
    short_name="i"
)
flags.DEFINE_integer(
    "steps",  # name of the parameter
    1500,  # Good balance between results and waiting time
    "Number of steps for neuron optimization, default 1500",
    short_name="s",
    lower_bound = 100
)
flags.DEFINE_float(
    "step_size",  # name of the parameter
    0.05,  # Good balance between results and waiting time
    "(0,1) float as step size, default 0.01",
    short_name="sz",
    lower_bound=0.000005,
    upper_bound=0.5
)
flags.DEFINE_integer(
    "save_every",  # name of the parameter
    100,  # Good balance between results and waiting time
    "[0-steps] int, steps after which an intermediate image is saved",
    short_name="se",
    lower_bound= 10
)
flags.DEFINE_spaceseplist(
    "file_name",  # name of the parameter
    "deep_dream_test",  # Good balance between results and waiting time
    "file name for saving images",
    short_name="fn"
)
# Regularizations
flags.DEFINE_float(
    "total_variance",  # name of the parameter
    -0.000000025,  # Good balance between results and waiting time
    "[0-0.1] float, Total variance regularization default=-0.000000025",
    short_name="tv",
    lower_bound=-0.1,
    upper_bound=0.1
)
flags.DEFINE_float(
    "lasso_1",  # name of the parameter
    0.2,  # Good balance between results and waiting time
    "(0,0.99) float, lasso 1 regularization",
    short_name="l1",
    lower_bound=0.0,
    upper_bound=0.99
)
flags.DEFINE_integer(
    "lasso_2",  # name of the parameter
    2,  # Good balance between results and waiting time
    "(0,10) int, lasso 2 regularization",
    short_name="l2",
    lower_bound=0,
    upper_bound=10
)
flags.DEFINE_integer(
    "padding",  # name of the parameter
    1,  # Good balance between results and waiting time
    "(0,10) int, pad regularization",
    short_name="p",
    lower_bound=0,
    upper_bound=10
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
def run_deep_dream_simple(img, deepdream, steps=100, step_size=0.01, save_every=50,file_name = str(int(time.time())), tv=0, l1=0, l2=0, pad=0,
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
        if steps_remaining > save_every:
            run_steps = tf.constant(save_every)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size), tv=tf.constant(tv), l1=tf.constant(l1),
                              l2=tf.constant(l2), pad=tf.constant(pad))
        save(img=deprocess(img), path= os.path.join(os.path.curdir, "output_images", f"d_{file_name}_{step}.jpg"),
             step = step, anno=f"{na}, {param_anno}")
        print("Step {}, loss {}".format(step, loss), end="\r")
    result = deprocess(img)
    show(result, step=steps, anno=f"{na}, {param_anno}")
    save(img=result,
         path=os.path.join(os.path.curdir, "output_images", f"d_{file_name}_final.jpg"),
         step=steps,
         anno=f"{na}, {param_anno}")
    print("Saved and finished")

    return result
def main(argv):
    # argv structure:
    # 0 = file name, list of neurons


    # Set hyperparameters
    STEPS = CLI.steps
    STEP_SIZE = CLI.step_size
    SAVE_EVERY = CLI.save_every
    FILE_NAME = CLI.file_name
    TV = CLI.total_variance
    L1 = CLI.lasso_1
    L2 = CLI.lasso_2
    PAD = CLI.padding
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
    print(output_neurons[0].shape)
    time.sleep(3)
    layers = [eval(s) for s in CLI.neurons]

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
        #Annotations:
        na = NAMES_ANNO)


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass
