import os

DIR_PATH = os.path.dirname(os.getcwd())
if not "tensorflow-yolov4-tflite" in DIR_PATH:
    DIR_PATH = os.path.join(DIR_PATH, "tensorflow-yolov4-tflite")
MODEL_ORIGINAL_PATH = os.path.join(DIR_PATH, "checkpoints", "yolo4-416")
VIS_DIR_PATH = os.path.join(DIR_PATH,"feature_vis")
MODEL_PLUS_PATH = os.path.join(DIR_PATH, "checkpoints", "yolov4-plus")
TRAINED_MODEL_PLUS_PATH = os.path.join(DIR_PATH, "checkpoints", "yolov4-trained")
BUILD_NEW = False


def get_DIR_PATH():
    return DIR_PATH


def get_MODEL_PLUS_PATH():
    return MODEL_PLUS_PATH


def get_BUILD_NEW():
    return BUILD_NEW


def get_VIS_DIR_PATH():
    return VIS_DIR_PATH

def get_TRAINED_MODEL_PLUS_PATH():
    return TRAINED_MODEL_PLUS_PATH



