import os.path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Normalize an image
def deprocess(img):
    img = tf.clip_by_value(img, -1, 1)
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

# Display an image
def show(img, step=None, anno=""):
    plt.figure()
    plt.imshow(img)
    if step != None:
        plt.suptitle(f"Step {step}")
    else:
        plt.suptitle(f"Step unknown")
    plt.title(anno)
    plt.axis('off')
    plt.show()

# Save an image with annotations
def save(img, path, step="done", anno=""):
    plt.figure()
    plt.imshow(img)
    if step != None:
        plt.suptitle(f"Step {step}")
    else:
        plt.suptitle(f"Step unknown")
    plt.title(anno)
    plt.axis('off')
    plt.savefig(path)

# Regularization by transformation robustness
def crop_and_pad(img, pad, mode="REFLECT", seed=None):
    img = tf.convert_to_tensor(img)
    shape = tf.shape(img)
    crop_shape = tf.concat([shape[-3:-1] - pad * 2, shape[-1:]], 0)

    img = tf.image.random_crop(img, crop_shape, seed=seed)

    img = tf.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode="REFLECT")

    return img

def get_starting_point(type="grey"):
    if type == "grey" or type == "gray":
        return np.full((416, 416, 3), 0.5, dtype="float32"), "grey"
    elif type == "random":
        return np.random.rand(416, 416, 3).astype("float32"), "random"
    else:
        img = cv2.imread(type)
        img = cv2.resize(img, (416, 416)) /255.
        return img.astype("float32"), "image"


