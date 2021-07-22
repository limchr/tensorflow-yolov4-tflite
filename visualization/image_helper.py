from math import sqrt,floor,ceil
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def preprocess_img(path):
    img = np.asarray(Image.open(path))/255
    #img = img[:img.shape[0]//8*8,:img.shape[1]//8*8,:] # make the image size dividable by x (necessary for some nets)
    return img



def convert_to_float_images(imgs):
    min = imgs.min()
    max = imgs.max()
    return (imgs-min)/(max-min)


def save_image(x,dir='.',filename='img'):
    x = normalize_img(x)
    img = Image.fromarray(x)
    img.save(os.path.join(dir,filename+'.png'))


def save_image_wo_norm(img,dir='.',filename='img.jpg'):
    # ensure that img is a numpy array
    img = np.array(img)

    # convert to color image
    # if len(img.shape) == 2:
    #     cimg = np.ones((img.shape[0],img.shape[1],3))
    #     for i in range(3):
    #         cimg[:,:,i] = img
    #     img = cimg

    img = np.array(img*255,dtype=np.uint8)


    img = Image.fromarray(img)
    img.save(os.path.join(dir,filename))



def normalize_img(img, int_img=True, col_img=True):

    # ensure that img is a numpy array
    img = np.array(img)

    # convert to color image
    if len(img.shape) == 2 and col_img:
        cimg = np.ones((img.shape[0],img.shape[1],3))
        for i in range(3):
            cimg[:,:,i] = img
        img = cimg

    # normalizing
    img = img.astype(np.float)
    min = img.min()
    max = img.max()
    img = (img-min)/(max-min)

    # return int image if wanted
    if int_img:
        img = np.array(img*255,dtype=np.uint8)

    return img


def grid_image(imgs): # create a grid image from array of imgs x imgs x imgw x imgh
    nx, ny, w, h = imgs.shape
    img = np.zeros((nx * w, ny * h),dtype=np.float)

    for x in range(nx):
        for y in range(ny):
            img[x*w:(x+1)*w,y*h:(y+1)*h] = imgs[x,y]

    return img


def hstack_images(*img_tuple, scale_to_max_width = True):

    imgs = []

    for i in img_tuple:
        imgs.append(normalize_img(i))

    shapes = np.array([i.shape for i in imgs])
    maxh, maxw, maxc = shapes.max(axis=0)
    if scale_to_max_width:
        imgs = [np.array(Image.fromarray(img).resize((int(maxh*(img.shape[1]/img.shape[0])),maxh))) for img in imgs]
        #recalculate this
        shapes = np.array([i.shape for i in imgs])
        maxh, maxw, maxc = shapes.max(axis=0)

    heights = shapes[:, 0]
    widths = shapes[:, 1]
    cum_widths = 0
    sumw = widths.sum()
    collage = np.zeros((maxh,sumw,maxc))
    for img,height,width in zip(imgs,heights,widths):
        img = convert_to_float_images(img)
        h = floor((maxh-height)/2)
        collage[h:h+height,cum_widths:cum_widths+width,:] = img
        cum_widths += width
    return collage

def vstack_images(*imgs, scale_to_max_width = True):
    imgs = [np.rot90(img,axes=(0, 1)) for img in imgs]
    collage = hstack_images(*imgs, scale_to_max_width = scale_to_max_width)
    return np.rot90(collage,axes=(1, 0))


def visualize_overlay(img, mask):
    maskimg = Image.fromarray(mask*255)
    maskimg = maskimg.resize((img.shape[1],img.shape[0]))
    filter_mask = np.array(maskimg)/255
    filter_mask[filter_mask < 0] = 0

    res = img.copy()
    c_mult = [0, 1, 0]
    for i in range(3):
        res[:, :, i] = res[:, :, i] * (filter_mask * c_mult[i] + np.ones(filter_mask.shape) * 0.5)
    return res

def draw_bbox(image, bboxes, colors=None, labels=None, border_width=None, font_scale=0.5):
    image_h, image_w, _ = image.shape

    if colors is None:
        colors = [[255, 0, 0]] * len(bboxes)
    if labels is None:
        labels = [None] * len(bboxes)
    if border_width is None:
        border_width = ceil((image_h + image_w) / 500)


    for coords, col, label in zip(bboxes, colors, labels):
        c1, c2 = [coords[0], coords[1]], [coords[0] + coords[2], coords[1] + coords[3]]
        cv2.rectangle(image, tuple(c1), tuple(c2), col, border_width)

        if not label is None:
            t_size = cv2.getTextSize(label, 0, font_scale, thickness=border_width // 2)[0]
            c3 = [c1[0] + t_size[0], c1[1] - t_size[1] - 3]
            if c3[1] < 0:
                c3[1] = c3[1] + t_size[1]
                c1[1] = c1[1] + t_size[1]
            cv2.rectangle(image, tuple(c1), tuple(c3), col, -1)
            cv2.putText(image, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        border_width // 2, lineType=cv2.LINE_AA)
    return image


if __name__ == '__main__':
    from tensorflow.keras.preprocessing.image import load_img
    ppl = preprocess_img('ppl.jpg')

    mask = [[0.5,0.3],[0.2,0.8]]

    hppl = hstack_images(ppl,ppl,mask, scale_to_max_width= True)

    save_image(hppl)