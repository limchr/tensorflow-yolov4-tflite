from absl import app, flags, logging
from absl.flags import FLAGS
import os
from core.dataset import Dataset
import numpy as np

from visualization.helper import get_model


from visualization.visualization_helper import make_gradcam_heatmap

from visualization.visualization_helper import make_gradcam_heatmap
from visualization.image_helper import save_image_wo_norm, draw_bbox
from common.helper import setup_clean_directory
from visualization.helper import get_model, get_class_names
from visualization.parameter import imgs, out_path_shift
from visualization.image_helper import read_img_yolo, draw_bbox, save_image_wo_norm




def main(argv):
    # set up paths
    setup_clean_directory(out_path_shift)

    num_steps = 10  # num_steps for left right top bottom
    im_size = 850

    model = get_model()
    class_names = get_class_names()
    for img_i,img in enumerate(imgs):
        im = read_img_yolo(img,target_size=[im_size,im_size])

        # create path and save base image
        base_path = os.path.join(out_path_shift,str(img_i))
        setup_clean_directory(base_path)

        inds = [1,3,5]
        num_cells_arr = [52,26,13]
        labels = ['s','m','l']

        for i,l,num_cells in zip(inds,labels,num_cells_arr):


            import math
            cell_size = im_size / num_cells
            cropped_size = math.floor(im_size - 2 * cell_size)
            cropr = [math.floor((cell_size * 2) / (num_steps-1) * x) for x in range(num_steps)]


            img_path = os.path.join(base_path,str(num_cells))
            setup_clean_directory(img_path)
            for xi,x in enumerate(cropr):
                for yi,y in enumerate(cropr):

                    cropped = im[y:cropped_size+y, x:cropped_size+x, :]
                    import cv2
                    cropped_resized = cv2.resize(cropped, (416,416))
                    pred = model.predict(cropped_resized.reshape([1] + list(cropped_resized.shape)))
                    p = pred[i]

                    cs = int(416/num_cells)
                    for xx in range(num_cells):
                        for yy in range(num_cells):
                            cell = p[0, yy, xx, :, :]
                            confs = cell[:, 4]  # isolate c neuron
                            i_max_conf = np.argmax(confs)
                            winner = cell[i_max_conf]
                            conf = float(winner[4])
                            cropped_resized = draw_bbox(cropped_resized,[[xx*cs+1,yy*cs+1,cs-2,cs-2]], colors=[[1-conf, conf, 0.0]], border_width=1)

                    save_image_wo_norm(cropped_resized,img_path,'img_%.5d_%.5d.jpg' % (xi,yi) )







if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass