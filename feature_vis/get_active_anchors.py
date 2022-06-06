import os
import sys
from absl import app, flags, logging
import numpy as np
import tensorflow as tf
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
print(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from visualization.helper import get_model, get_class_names
from visualization.image_helper import read_img_yolo, draw_bbox, save_image_wo_norm


CLI = flags.FLAGS
flags.DEFINE_spaceseplist(
    "img_anchor",  # name of the parameter
    ["feature_vis", "data", "000000000431.jpg"], # Gray img
    "String of path to single input image to use via os.path, seperated by spaces, default is a gray image",
    short_name="i_a"
)
flags.DEFINE_float(
    "conf",  # name of the parameter
    0.25,
    "Expects float [0-1], ignores boxes below this confidence, default 0.25"
)

def get_anchors_for_img(img, conf):
    model = get_model()
    class_names = get_class_names()

    im = read_img_yolo(img)
    im_size = 416
    pred = model.predict(im.reshape([1] + list(im.shape)))
    inds = [1, 3, 5]
    labels = ['s', 'm', 'l']
    anchors_high_conf = []
    bboxes = []
    to_out = []
    for i, l in zip(inds, labels):
        p = pred[i]
        num_cells = p.shape[1]
        # setup_clean_directory(img_path)
        for x in range(num_cells):
            for y in range(num_cells):
                skip = True
                for w in range(3):
                    c = p[0, y, x, w, 4]
                    if c >= CLI.conf:
                        anchors_high_conf.append([i, 0, y, x, w])
                        bboxes.append(p[0, y, x, w])

        re_bbox = np.asarray(bboxes)[:, 0:4]
        re_bbox = np.reshape(re_bbox, (1, re_bbox.shape[0], 1, 4))
        re_conf = np.asarray(bboxes)[:, 5:]
        re_conf = np.reshape(re_conf, (1, re_conf.shape[0], 80))

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=re_bbox,
            scores=re_conf,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25,
            clip_boxes=False
        )
        boxes = boxes.numpy()[0]
        valid_detections = valid_detections.numpy()[0]
        re_bbox = np.reshape(re_bbox, (re_bbox.shape[1], 4))
        former_indices = []

        for i in range(valid_detections):
            index = np.where(re_bbox == boxes[i])[0]
            former_indices.append(index[0])
            p = f"bbox: {boxes[i]}\nclass: {class_names[classes.numpy()[0][i]]}\nscore: {scores.numpy()[0][i]}\nanchorbox: {anchors_high_conf[former_indices[-1]]}\n\n"
            to_out.append(f"output_neurons[{anchors_high_conf[former_indices[-1]][0]}]"
                          f"[{anchors_high_conf[former_indices[-1]][1]}]"
                          f"[{anchors_high_conf[former_indices[-1]][2]}]"
                          f"[{anchors_high_conf[former_indices[-1]][3]}]"
                          f"[{anchors_high_conf[former_indices[-1]][4]}]"
                          f"[{int(classes.numpy()[0][i]) + 5}]")
            print(f"{len(to_out)}:\n{p}")
    return to_out, model


def main(argv):
    image = os.path.join(*CLI.img_anchor)
    conf = CLI.conf
    all_anchors = get_anchors_for_img(image,conf)
    print(all_anchors)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass