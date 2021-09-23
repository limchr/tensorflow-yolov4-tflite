import os
import matplotlib.pyplot as plt
import numpy as np

from common.helper import setup_clean_directory


from absl import app, flags, logging
from absl.flags import FLAGS

from visualization.helper import get_model, get_class_names
from visualization.parameter import imgs, out_path_bboxes
from visualization.image_helper import read_img_yolo, draw_bbox, save_image_wo_norm


def main(argv):
    # set up paths
    setup_clean_directory(out_path_bboxes)


    model = get_model()
    class_names = get_class_names()
    for img_i,img in enumerate(imgs):
        im = read_img_yolo(img)

        # create path and save base image
        base_path = os.path.join(out_path_bboxes,str(img_i))
        setup_clean_directory(base_path)
        save_image_wo_norm(im,base_path,'base.jpg')

        im_size = 416
        pred = model.predict(im.reshape([1] + list(im.shape)))
        inds = [1,3,5]
        labels = ['s','m','l']

        for i,l in zip(inds,labels):
            p = pred[i]
            num_cells = p.shape[1]
            img_path = os.path.join(base_path,str(num_cells))
            setup_clean_directory(img_path)
            for x in range(num_cells):
                for y in range(num_cells):
                    cell_size = int(im_size / num_cells)
                    imd = im.copy()
                    cellxywh = [cell_size * x, cell_size * y, cell_size, cell_size]
                    imd = draw_bbox(imd, [cellxywh], colors=[[0.0, 0.0, 1.0]])

                    # use most certain anchorbox
                    cell = p[0, y, x, :, :]
                    confs = cell[:, 4] # isolate c neuron
                    i_max_conf = np.argmax(confs)
                    winner = cell[i_max_conf]
                    conf = float(winner[4])
                    xywh = np.array(winner[:4], dtype=np.int)
                    xywh[0] -= xywh[2] / 2
                    xywh[1] -= xywh[3] / 2
                    cl_i = np.argmax(winner[5:])

                    imd = draw_bbox(imd, [xywh], colors=[[1-conf, conf, 0.0]],
                                    labels=['%s(c:%.2f;a:%d)' % (class_names[cl_i], conf, i_max_conf)], font_scale=0.7)
                    save_image_wo_norm(imd,img_path,'img_%.5d_%.5d.jpg' % (x,y) )



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass