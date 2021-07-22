from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import math

from image_helper import draw_bbox

import matplotlib.pyplot as plt
import cv2




flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './data/yolov4.weights', 'pretrained weights')
# flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

flags.DEFINE_string('image_path', './input_images', 'path to input image')
flags.DEFINE_string('output_path', './output_images', 'path to output image')



def receptive_field(model, image_data, target, img_num, class_names,p_filter=None):
    image_data = image_data[[0], :, :, :]

    # save original image

    from image_helper import save_image_wo_norm
    from common.helper import setup_clean_directory
    from PIL import Image

    # RECEPTIVE FIELD COMPUTATION

    # mask_ratio_x = [0.2,0.4,0.6,0.8]
    # mask_ratio_y = [0.2,0.4,0.6,0.8]

    mask_ratio_x = [1.0 / 13.0 * i for i in range(13)]
    mask_ratio_y = [1.0 / 13.0 * i for i in range(13)]

    mask_selection = [[True,False,False],[False,True,False],[False,False,True]]
    # mask_selection = [[False, False, True],[True, False, False]]
    # mask_selection = [[False, False, True]]

    mask_selection_labels = 'sml'

    mask_grid_selection = [[True, False, False, False, False, False],
                           [False, True, False, False, False, False],
                           [False, False, True, False, False, False],
                           [False, False, False, True, False, False],
                           [False, False, False, False, True, False],
                           [False, False, False, False, False, True],
                           ]
                           #[True, True, True, True, True, True]]
    # mask_grid_selection = [[False, False, False, False, False, True]]
    mask_grid_labels = 'xywhcp'

    sub_img_num = 0
    grad_threshold = 0.01
    file_type = '.jpg'

    plt_dir = 'figs'
    img_dir = os.path.join(plt_dir, str(img_num))

    setup_clean_directory(img_dir)
    # setup_clean_directory(os.path.join(img_dir, 'combined'))
    # setup_clean_directory(os.path.join(img_dir, 'norm-grad'))
    setup_clean_directory(os.path.join(img_dir, 'static-grad'))
    # setup_clean_directory(os.path.join(img_dir, 'thresh-grad'))
    setup_clean_directory(os.path.join(img_dir, 'bbs'))
    setup_clean_directory(os.path.join(img_dir, 'static-grad-pn'))
    setup_clean_directory(os.path.join(img_dir, 'grad-cam'))

    save_image_wo_norm(image_data[0], img_dir, 'base_img'+file_type)

    if not p_filter is None:
        with open(os.path.join(img_dir,'p_filter_class_names.txt'),'w') as f:
            f.write(','.join(p_filter))


    for mask_sel in mask_selection:
        mask_sel_str = ''.join([mask_selection_labels[i] if e else '-' for i, e in enumerate(mask_sel)])
        for mask_grid_sel in mask_grid_selection:
            mask_grid_str = ''.join([mask_grid_labels[i] if e else '-' for i, e in enumerate(mask_grid_sel)])

            for mask_rat_x in mask_ratio_x:
                for mask_rat_y in mask_ratio_y:
                    input = tf.convert_to_tensor(image_data)
                    losses = []
                    with tf.GradientTape() as tf_gradient_tape:
                        tf_gradient_tape.watch(input)
                        # get the predictions
                        preds = model(input)

                        for mask_ind in range(3):
                            if mask_sel[mask_ind]:
                                pred_i = mask_ind * 2 + 1
                                mask1 = np.zeros(preds[pred_i].shape, dtype=np.float)

                                grid_cell_mask = np.zeros(mask1.shape[-1], dtype=np.bool)
                                grid_cell_mask[0] = True if mask_grid_sel[0] else False  # x
                                grid_cell_mask[1] = True if mask_grid_sel[1] else False  # y
                                grid_cell_mask[2] = True if mask_grid_sel[2] else False  # w
                                grid_cell_mask[3] = True if mask_grid_sel[3] else False  # h
                                grid_cell_mask[4] = True if mask_grid_sel[4] else False  # confidence
                                if mask_grid_sel[5]:
                                    if p_filter is None:
                                        grid_cell_mask[5:] = True  # class probabilities
                                    else:
                                        cl_ind = list(class_names.keys())[list(class_names.values()).index(p_filter[0])]
                                        grid_cell_mask[5+cl_ind] = True
                                mask1[:, int(mask1.shape[1] * mask_rat_y), int(mask1.shape[2] * mask_rat_x), :,
                                grid_cell_mask] = 1
                                # mask1[:, 0, 0, :] = 1
                                mask1_tensor = tf.convert_to_tensor(mask1, dtype=tf.float32)
                                mask1_output = mask1_tensor * preds[pred_i]
                                loss1 = tf.reduce_sum(mask1_output/tf.math.reduce_sum(mask1_output))
                                losses.append(loss1)






                        if len(losses) > 1:
                            pseudo_loss = tf.add(*losses)
                        else:
                            pseudo_loss = losses[0]

                        # get gradient
                        img = image_data[0].copy()
                        grad = tf_gradient_tape.gradient(pseudo_loss, input)[0].numpy()

                        # variables for plotting
                        normalized_gradients = np.abs(grad)
                        normalized_gradients = normalized_gradients / normalized_gradients.max()

                        thresholded_gradients = np.array(normalized_gradients > grad_threshold, dtype=np.float)


                        fixed_normalization_factor = 1000000000
                        fixed_normalization = np.abs(grad * fixed_normalization_factor)
                        fixed_normalization[fixed_normalization > 1.0] = 1.0

                        grad_1d = grad.mean(axis=2) * fixed_normalization_factor
                        grad_1d_p = grad_1d >= 0
                        grad_1d_n = grad_1d < 0
                        grad_pn = np.zeros(grad.shape, dtype=np.float)

                        grad_pn[grad_1d_p, 0] = np.abs(grad_1d)[grad_1d_p]
                        grad_pn[grad_1d_n, 2] = np.abs(grad_1d)[grad_1d_n]


                        for mask_ind in range(3):
                            if mask_sel[mask_ind]:
                                pred_i = mask_ind * 2 + 1
                                pred = preds[pred_i]
                                num_cells = pred.shape[1]
                                bb_cell_i = int(num_cells * mask_rat_x), int(num_cells * mask_rat_y)
                                cell_dim = grad.shape[0] // num_cells
                                cellxywh = [cell_dim * bb_cell_i[0], cell_dim * bb_cell_i[1], cell_dim, cell_dim]
                                img = draw_bbox(img, [cellxywh])
                                normalized_gradients = draw_bbox(normalized_gradients, [cellxywh])
                                fixed_normalization = draw_bbox(fixed_normalization, [cellxywh])
                                thresholded_gradients = draw_bbox(thresholded_gradients, [cellxywh])

                                # cell bounding box visualization
                                bb_cell = pred[0, bb_cell_i[1], bb_cell_i[0], :, :]

                                bb_conf = bb_cell[:, 4]
                                i_max_conf = np.argmax(bb_conf)

                                bb = bb_cell[i_max_conf]
                                conf = float(bb[4])
                                xywh = np.array(bb[:4], dtype=np.int)
                                xywh[0] -= xywh[2] / 2
                                xywh[1] -= xywh[3] / 2

                                cl_i = np.argmax(bb[5:])

                                img = draw_bbox(img, [xywh], colors=[[0.0, 0.0, 1.0]], labels=['%s C: %.2f BB_i: %d' % (class_names[cl_i],conf,i_max_conf)], font_scale=0.7)



                                # for bb in bb_cell:
                                #     conf = float(bb[4])
                                #     xywh = np.array(bb[:4], dtype=np.int)
                                #     #xywh[0] -= xywh[2] / 2
                                #     #xywh[1] -= xywh[3] / 2
                                #     img = draw_bbox(img, [xywh], colors=[[0.0, 0.0, 1.0]], labels=['Conf: %.2f' % (conf,)])

                                # preds[pred_i][0, , 0, :4]

                                name_last_backbone_conv = 'conv2d_77'
                                name_last_backbone_conv = 'conv2d_50'
                                name_last_backbone_conv = 'conv2d_20'
                                backprop_i = np.where(mask_grid_sel)[0][0]
                                grad_cam = make_gradcam_heatmap(image_data, model, name_last_backbone_conv, pred_i, bb_cell_i[1],
                                                     bb_cell_i[0], i_max_conf, backprop_i)
                        base_filename_append = mask_sel_str + '_' + mask_grid_str + '_' + (
                                    '%.5d' % (bb_cell_i[0],)) + '_' + ('%.5d' % (bb_cell_i[1],))

                        if False:
                            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                            ax1.imshow(img)
                            ax2.imshow(normalized_gradients)
                            ax3.imshow(fixed_normalization)
                            ax4.imshow(thresholded_gradients)
                            ax1.set_title('input image')
                            ax2.set_title('normalized gradients (0-1)')
                            ax3.set_title('static normalization')
                            ax4.set_title('thresholded gradients (>' + str(grad_threshold) + ')')
                            ax1.get_xaxis().set_visible(False)
                            ax1.get_yaxis().set_visible(False)
                            ax2.get_xaxis().set_visible(False)
                            ax2.get_yaxis().set_visible(False)
                            ax3.get_xaxis().set_visible(False)
                            ax3.get_yaxis().set_visible(False)
                            ax4.get_xaxis().set_visible(False)
                            ax4.get_yaxis().set_visible(False)
                            fig.tight_layout()
                            plt.savefig(os.path.join(img_dir, 'combined', base_filename_append + file_type))
                            plt.close(fig)
                        # plt.show()

                        # save_image_wo_norm(normalized_gradients, os.path.join(img_dir, 'norm-grad'), base_filename_append + file_type)
                        save_image_wo_norm(fixed_normalization, os.path.join(img_dir, 'static-grad'), base_filename_append + file_type)
                        save_image_wo_norm(grad_pn, os.path.join(img_dir, 'static-grad-pn'), base_filename_append + file_type)
                        save_image_wo_norm(grad_cam, os.path.join(img_dir, 'grad-cam'), base_filename_append + file_type)


                        # save_image_wo_norm(thresholded_gradients, os.path.join(img_dir, 'thresh-grad'), base_filename_append + file_type)
                        if mask_grid_str == 'x-----':
                            save_image_wo_norm(img, os.path.join(img_dir, 'bbs'), base_filename_append + file_type)

                        sub_img_num += 1


def get_model():
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)

    return model





def main(_argv):

    trainset = Dataset(FLAGS, is_training=False)
    class_names = trainset.classes

    model = get_model()


    trainset = [trainset.__next__() for i in range(5)]




    p_filter = [
        ['sandwich','spoon'],
        ['laptop', 'cat'],
        ['person', 'skateboard'],
        ['cat', 'bed'],
        ['pizza', 'dining table'],
    ]


    for i in range(len(trainset)):
        image_data = trainset[i][0]
        target = trainset[i][1]
        pf = p_filter[i]
        receptive_field(model, image_data, target, i, class_names,p_filter=pf)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass