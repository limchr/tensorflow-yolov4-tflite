import os.path
import sys
import subprocess
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from visualization.helper import get_model, get_class_names
from feature_vis.approach_lucid import deep_dream_with_anchors as deep_dream
from feature_vis.get_active_anchors import get_anchors_for_img as get_anchors
from absl import app, flags


CLI = flags.FLAGS

def main(argv):
    img = os.path.join(*CLI.img_anchor)
    anchors, model = get_anchors(img, CLI.conf)
    deep_dream(neurons=anchors, img=img, steps=CLI.steps, step_size=CLI.step_size,
               save_every=CLI.save_every, file_name=CLI.file_name, file_path=CLI.file_path,
               tv=CLI.total_variance, l1=CLI.lasso_1, l2=CLI.lasso_2,
               pad=CLI.padding, c=CLI.color_difference, reproduce=CLI.reproduce, model = model)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
