# coding=utf-8
"""
@File   : preprocess.py
@Time   : 2020/04/14
@Author : Zengrui Zhao
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure
from skimage.filters import threshold_otsu
from PIL import Image

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)

def binarize(image):
    thres = threshold_otsu(image)
    bw = image <= thres

    label = measure.label(bw)
    properties = measure.regionprops(label)
    valid_label = []
    shape = image.shape
    limit = [.1, .9]
    for prop in properties:
        centroid = np.array(prop.centroid) / np.array(shape)
        if .4 < centroid[1] < .5 and (centroid[0] < .3 or centroid[0] > .7): # 除掉上下区域
            continue
        if prop.convex_area / shape[0] / shape[1] > .8: # 除掉超大区域
            continue
        if prop.convex_area / prop.area > 2.3: # 除掉周围黑色的大块区域(2和3之间)
            continue
        if abs(centroid[0] - centroid[1]) > .6: # 除掉左下角和右上角区域
            continue
        if (centroid[0] > .8 and centroid[1] > .8) or (centroid[0] < .3 and centroid[1] < .3): # 除掉左上角和右下角区域
            continue
        x0, y0, x1, y1 = prop.bbox
        if (x1 - x0 == shape[0] and y1 - y0 > shape[1] / 2) or (y1 - y0 == shape[1] and x1 - x0 > shape[0] / 2): # 除掉超大快
            continue
        if (y1 - y0) / shape[1] > .8: # 一般不会出现这种情况
            continue

        if limit[0] < centroid[0] < limit[1] and limit[0] < centroid[1] < limit[1]:
            valid_label.append(prop.label)

    bw = np.in1d(label, valid_label).reshape(label.shape)
    return bw

def extractLung(bw):
    label = measure.label(bw)
    properties = measure.regionprops(label)
    properties.sort(key=lambda x: x.area, reverse=True)
    properties = [prop for prop in properties[:2] if prop.bbox_area / bw.size > .06]
    if len(properties) == 2:
        x0, y0, x1, y1 = properties[0].bbox
        x3, y3, x4, y4 = properties[1].bbox
        iou = compute_IOU((x0, y0, x1, y1), (x3, y3, x4, y4))
        if iou > .1:
            print(iou)
            properties.pop(-1)

    valid_label = [prop.label for prop in properties]
    bw = np.in1d(label, valid_label).reshape(label.shape)

    return bw

def boundingBox(bw, image):
    label = measure.label(bw)
    properties = measure.regionprops(label)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for prop in properties:
        minr, minc, maxr, maxc = prop.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)
    plt.show()

def cropImg(bw, image, path, name):
    outputPath = os.path.join('/home/zzr/Data/XinGuan/lung', '/'.join(path.split('/')[-2:]))
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    image = Image.fromarray(image)
    label = measure.label(bw)
    properties = measure.regionprops(label)
    for idx, prop in enumerate(properties):
        minr, minc, maxr, maxc = prop.bbox
        output = image.crop((minc, minr, maxc, maxr))
        # plt.imshow(output, cmap='gray')
        # plt.show()
        outPath = os.path.join(outputPath, f'{name}_{str(idx)}.png')
        output.save(outPath)

def main():
    path = '/home/zzr/Data/XinGuan/data/test/COVID'
    imgs = list(map(lambda img: os.path.join(path, img), sorted(os.listdir(path))))
    for idx, img in enumerate(imgs):
        if idx >= 0:
            name = img.split("/")[-1]
            print(f'{idx} / {len(imgs)}--{name}')
            data = np.array(Image.open(img).convert('L'))
            bw = binarize(data)
            try:
                bw = extractLung(bw)
                boundingBox(bw, data)
                # cropImg(bw, data, path, name)
            except:
                print(img)

if __name__ == '__main__':
    main()
