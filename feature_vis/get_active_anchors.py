import os
import sys
from absl import app, flags, logging
import numpy as np
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
print(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from visualization.helper import get_model, get_class_names
from visualization.image_helper import read_img_yolo, draw_bbox, save_image_wo_norm


def main(argv):
    model = get_model()
    class_names = get_class_names()
    im = read_img_yolo("data/000000000431.jpg")
    im_size = 416
    pred = model.predict(im.reshape([1] + list(im.shape)))
    inds = [1,3,5]
    labels = ['s','m','l']
    results = []
    conf_bbox = []

    for i,l in zip(inds,labels):
        p = pred[i]
        num_cells = p.shape[1]
        for x in range(num_cells):
            for y in range(num_cells):
                for i in range(3):
                    c = p[0, y, x, i, 4]
                    if c < 0.25: continue
                cell_size = int(im_size / num_cells)
                imd = im.copy()
                cellxywh = [cell_size * x, cell_size * y, cell_size, cell_size]
                imd = draw_bbox(imd, [cellxywh], colors=[[0.0, .45, .25]])

                # use most certain anchorbox
                cell = p[0, y, x, :, :]

                confs = cell[:, 4] # isolate c neuron
                i_max_conf = np.argmax(confs)
                winner = cell[i_max_conf]
                conf = float(winner[4])
                # x and y are index of winner anchor?
                xywh = np.array(winner[:4], dtype=np.int)

                xywh[0] -= xywh[2] / 2
                xywh[1] -= xywh[3] / 2
                cl_i = np.argmax(winner[5:])
                results.append((i,l,xywh,np.argmax(winner[5:]),conf, class_names[cl_i]))
                imd = draw_bbox(imd, [xywh], colors=[[1-conf, 0.0, conf]],
                                labels=['%s(c:%.2f;a:%d)' % (class_names[cl_i], conf, i_max_conf)], font_scale=0.7)
                save_image_wo_norm(imd, img_path, 'img_%.5d_%.5d.jpg' % (x, y))
    print(results)
    print(conf_bbox)

    return results

def main_2(argv):
    # set up paths
    #setup_clean_directory(out_path_bboxes)


    model = get_model()
    class_names = get_class_names()
    imgs = ["/Users/fabianreichwald/Documents/tensorflow-yolov4-tflite/feature_vis/data/000000000431.jpg"]
    for img_i,img in enumerate(imgs):
        im = read_img_yolo(img)

        # create path and save base image
        base_path = os.path.join(out_path_bboxes,str(img_i))
        # setup_clean_directory(base_path)
        save_image_wo_norm(im,base_path,'base.jpg')

        im_size = 416
        pred = model.predict(im.reshape([1] + list(im.shape)))
        inds = [1,3,5]
        labels = ['s','m','l']
        for i,l in zip(inds,labels):
            p = pred[i]
            num_cells = p.shape[1]
            img_path = os.path.join(base_path,str(num_cells))
            # setup_clean_directory(img_path)
            for x in range(num_cells):
                for y in range(num_cells):
                    for i in range(3):
                        c = p[0, y, x, i, 4]
                        if c < 0.25: continue
                    cell_size = int(im_size / num_cells)
                    imd = im.copy()
                    cellxywh = [cell_size * x, cell_size * y, cell_size, cell_size]
                    imd = draw_bbox(imd, [cellxywh], colors=[[0.0, .45, .25]])

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

                    imd = draw_bbox(imd, [xywh], colors=[[1-conf, 0.0, conf]],
                                    labels=['%s(c:%.2f;a:%d)' % (class_names[cl_i], conf, i_max_conf)], font_scale=0.7)
                    save_image_wo_norm(imd,img_path,'img_%.5d_%.5d.jpg' % (x,y) )


if __name__ == '__main__':
    try:
        app.run(main_2)
    except SystemExit:
        pass