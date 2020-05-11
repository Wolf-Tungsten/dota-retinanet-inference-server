#!/usr/bin/env python
# coding: utf-8

import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# ## Load necessary modules
import tensorflow as tf
import keras

import sys

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
# from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import csv
import numpy as np
import time
import math
import random
# import argparse


# set tf backend to allow memory to grow, instead of claiming everything

# 启用 gpu 0号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


# 目标检测算法输入窗口大小
MODEL_SIZE = 224

# 默认输出目录，输出经过标注的图片
DEFAULT_OUTPUT_DIR = r'./tmp'

# 模型文件路径
MODEL_PATH = r'/root/train/resnet50_csv_50.h5'

# load retinanet model
model = models.load_model(MODEL_PATH, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

# print(model.summary())

labels_to_names = {0: 'small-vehicle', 1: 'plane', 2: 'harbor'}


class Image:
    '''本脚本中 img 一律指 Image 类的实例， cvimg 才指代 opencv 图片实例。'''

    def __init__(self,
                 cvimg,                     # OpenCV 图像
                 filename=None,             # 文件名，不含路径部分
                 type="original",           # "original": 原图, "small": 1.0 缩放,
                                            # "middle": 0.7 缩放, "large": 0.4 缩放
                 posX=0, posY=0,            # 相对原图位移
                 annotation=None,             # 经过推理，对图片中目标的标注
                                            # [(x1, y1, x2, y2, scroe, type), ...]
                 original_image=None):      # 对原图 Image 实例的引用

        self.id = Image.id_counter
        Image.id_counter += 1

        self.cvimg = cvimg
        self.width = cvimg.shape[1]
        self.height = cvimg.shape[0]

        self.filename = filename
        self.type = type
        self.posX = posX
        self.posY = posY
        self.annotation = [] if annotation == None else annotation
        self.original_image = original_image


Image.id_counter = 0


def handle(image_path, output_image_dir=DEFAULT_OUTPUT_DIR, output_annotation_dir=''):
    print('当前处理 ' + image_path + '...')

    cvimg = cv2.imread(image_path)
    img = Image(
        cvimg=cvimg,
        filename=image_path.split('/').pop().split('\\').pop(),
    )

    # 切割图片
    print('  1. 切割图片中...', end='\r')
    [small_imgs, middle_imgs, large_imgs] = split_image(
        img,
        output_split_images='./tmp/split'  # 去注释该行，可以输出分割后未推理的图片
    )
    print('  1. 切割图片    √')

    split_imgs = small_imgs + middle_imgs + large_imgs
    split_imgs_len = len(split_imgs)
    raw_anno = []

    for i, simg in enumerate(split_imgs):
        raw_anno += infer(simg)

        # 去注释下面函数，可以输出分割后经推理的图片
        output_split_images_with_annotation(
            simg,
            './tmp/split-anno'
        )

        print('  2. 推理中... {}/{}'.format(i + 1, split_imgs_len), end='\r')

    print(' '*40, end='\r')
    print('  2. 推理        √')

    print('  3. 合并中...', end='\r')
    img.annotation = merge(img, raw_anno)
    print('  3. 合并        √')
    print()

    cvimg = cvimg.copy()
    for x1, y1, x2, y2, score, kind in img.annotation:
        cv2.rectangle(cvimg, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(
        os.path.join(output_image_dir, img.filename + '-anno.png'),
        cvimg
    )

    if type(output_annotation_dir) == str and output_annotation_dir != '':
        f = open(os.path.join(output_annotation_dir, img.filename + '-anno.csv'), 'w')
        f_csv = csv.writer(f)
        f_csv.writerow(['x1', 'y1', 'x2', 'y2', 'score', 'type'])
        f_csv.writerows(img.annotation)
        f.close()



def merge(img: Image, raw_anno):
    # raw_anno: [(x1, y1, x2, y2, score, type), ...]

    result = []
    anno = []

    for raw_anno_item in raw_anno:
        anno.append([
            (raw_anno_item[0] + raw_anno_item[1]) * 0.5,    # x坐标平均
            raw_anno_item[5],                               # type
            False,                                          # 该项是否已被考虑
            raw_anno_item                                   # 原始项
        ])

    anno.sort(key=lambda x: (x[1], x[0]))  # 首先按 type 字母排序，再按 x坐标平均值排序

    for anno_base in anno:
        if anno_base[2]:  # 若该项已被考虑过，跳过
            continue

        anno_base[2] = True
        (base_x_c, _, base_considered, base_item) = anno_base
        (base_x1, base_y1, base_x2, base_y2, base_scroe, base_kind) = base_item

        overlap_targets = [base_item]

        base_S = (base_x2 - base_x1) * (base_y2 - base_y1)
        x_right_lim = 1.5 * base_x2 - 0.5 * base_x1
        for anno_test in anno:
            (x_c, _, considered, item) = anno_test
            (x1, y1, x2, y2, scroe, kind) = item

            if kind > base_kind or x_c >= x_right_lim:
                break

            if (kind != base_kind) or considered or (anno_base == anno_test):
                continue

            if x2 <= base_x1 or x1 >= base_x2 or y2 <= base_y1 or y1 >= base_y2:
                continue

            overlap_x = x2 - x1 + base_x2 - base_x1 - \
                (max(x2, base_x2) - min(x1, base_x1))
            overlap_y = y2 - y1 + base_y2 - base_y1 - \
                (max(y2, base_y2) - min(y1, base_y1))
            overlap_S = overlap_x * overlap_y
            overlap_rate = max(
                overlap_S / base_S,
                overlap_S / ((x2 - x1) * (y2 - y1))
            )
            if overlap_rate >= 0.5:
                anno_test[2] = True
                overlap_targets.append(item)

        if len(overlap_targets) == 1:
            result.append(base_item)
        else:
            x1_c = 0.
            y1_c = 0.
            x2_c = 0.
            y2_c = 0.
            score_c = 0.

            for x1, y1, x2, y2, score, kind in overlap_targets:
                x1_c += x1
                y1_c += y1
                x2_c += x2
                y2_c += y2
                score_c += score

            x1_c /= len(overlap_targets)
            y1_c /= len(overlap_targets)
            x2_c /= len(overlap_targets)
            y2_c /= len(overlap_targets)
            score_c /= len(overlap_targets)

            result.append((int(x1_c), int(y1_c),
                           int(x2_c), int(y2_c), score_c, type))

    return result


def output_split_images_with_annotation(img, path):
    cvimg = img.cvimg.copy()
    for (x1, y1, x2, y2, score, type) in img.annotation:
        cv2.rectangle(cvimg, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(path, img.filename + '-anno.png'), cvimg)


def split_image(img, output_split_images=False):
    filename_without_extension = None \
        if img.filename == None \
        else img.filename.split('.')[0]

    small_size = int(MODEL_SIZE)
    middle_size = int(MODEL_SIZE / 0.7)
    large_size = int(MODEL_SIZE / 0.4)

    sliding_step_ratio = 0.5  # 滑动窗口步长 = 0.5 倍窗口大小

    size_names = ['small', 'middle', 'large']
    sizes = [small_size, middle_size, large_size]

    result = [[], [], []]

    for i in range(3):
        size = sizes[i]
        size_name = size_names[i]
        sliding_step = int(size * sliding_step_ratio)

        x_ith = 0
        y_ith = 0

        posX = 0
        posY = 0

        reach_x_end = False
        reach_y_end = False
        while True:
            if posY + size >= img.height:
                reach_y_end = True
                posY = img.height - size
                if posY < 0:
                    posY = 0

            x_ith = 0
            posX = 0
            reach_x_end = False

            while True:
                if posX + size >= img.width:
                    reach_x_end = True
                    posX = img.width - size
                    if posX < 0:
                        posX = 0

                posY2 = img.height if reach_y_end else posY + size
                posX2 = img.width if reach_x_end else posX + size

                img_slice_filename = None
                if filename_without_extension != None:
                    img_slice_filename = filename_without_extension + '-' + \
                        size_name + '-y' + str(y_ith) + \
                        '-x' + str(x_ith) + '.png'
                img_slice = Image(
                    cvimg=img.cvimg[posY:posY2, posX:posX2],
                    filename=img_slice_filename,
                    type=size_name,
                    posX=posX,
                    posY=posY,
                    original_image=img
                )

                result[i].append(img_slice)

                if type(output_split_images) == str:
                    cv2.imwrite(os.path.join(output_split_images,
                                             img_slice_filename), img_slice.cvimg)

                posX += sliding_step
                x_ith += 1
                if reach_x_end:
                    break

            posY += sliding_step
            y_ith += 1
            if reach_y_end:
                break

    return result


def infer(img):
    bgr_image = cv2.cvtColor(img.cvimg, cv2.COLOR_RGB2BGR)
    bgr_image = preprocess_image(bgr_image)
    bgr_image, scale = resize_image(bgr_image)

    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(bgr_image, axis=0))

    # correct for image scale
    boxes /= scale
    result = []
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        box = box.astype(int)
        box[0] = 0 if box[0] < 0 else box[0]
        box[1] = 0 if box[1] < 0 else box[1]
        box[2] = 0 if box[2] < 0 else box[2]
        box[3] = 0 if box[3] < 0 else box[3]
        img.annotation.append((
            box[0], box[1],
            box[2], box[3],
            score,
            labels_to_names[label]
        ))
        result.append((
            box[0] + img.posX, box[1] + img.posY,
            box[2] + img.posX, box[3] + img.posY,
            score,
            labels_to_names[label]
        ))

    return result


handle('./images/images/P0000.png')
