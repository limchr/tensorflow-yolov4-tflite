import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
from ipywidgets.widgets import interact, fixed
import matplotlib.animation as animation
#import seaborn as ssns

import feature_vis.config as config
import os
import cv2
import itertools
import pandas as pd

l1 = [0, 0.2, 2]
l2 = [0, 0.2, 2]
pad = [0,1,2,5, 10]
tv = [0, -0.000000025, -0.000001]
s = [l1, l2, pad, tv]
choices = list(itertools.product(*s))
configs = []
i = 0
for n in choices:
	if n[0] + n[1] == 0:
		continue
	configs.append([i, n[0], n[1], n[2], n[3], 2000])# l1, l2, pad, tv, steps
	i += 1

def find_config(l1, l2, pad, tv):
	for config in configs:
		if config[1] == l1 and config[2] == l2 and config[3] == pad and config[4] == tv:
			return config[0]
	return -1

def plot(l1=0.2, l2=0, pad=0, tv=0):
	fig, ax = plt.subplots(figsize=(24, 24))
	i = find_config(l1, l2, pad, tv)
	img = cv2.imread(os.path.join("feature_vis", "output_images", f"'abl_{i}'", f"'{i}'_final.png"))
	ax.imshow(img)

#interactive(plot, l1=[0, 0.2, 2], l2=[0, 0.2, 2], pad=[0, 1, 2, 5, 10], tv=[0, -0.000000025, -0.000001])
def plot_grid(pad, tv):
	fig, axes = plt.subplots(3, 3, figsize=(25, 25))
	for y, ax_y in enumerate(axes):
		for x, ax in enumerate(ax_y):
			ax.grid(False)
			ax.set_xticks([])
			ax.set_yticks([])

			i = find_config(l1[y], l2[x], pad, tv)
			if i != -1:
				path = os.path.join(config.get_DIR_PATH(), "output_images", f"'abl_{i}'", f"'{i}'_final.png")
				img = cv2.imread(path)
				ax.imshow(img)
	plt.subplots_adjust(wspace=0, hspace=-0.5)
	plt.show()

plot_grid(1, tv[1])


