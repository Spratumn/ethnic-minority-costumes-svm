import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.feature
import skimage.segmentation
import time
import math

def get_lbp_features(image, cellNums=(4, 4), binSize=64, flag=0, normal=False):
    lbp = extract_basic_lbp(image=image, flag=flag)
    h, w = lbp.shape[:2]
    cellSizeH = h // cellNums[1]
    cellSizeW = w // cellNums[0]
    lbpHist = []
    for i in range(cellNums[0]):
        for j in range(cellNums[1]):
            l = i*cellSizeW 
            r = w if i == cellNums[0] - 1 else (i+1)*cellSizeW 
            t = j*cellSizeH 
            b = h if j == cellNums[1] - 1 else (j+1)*cellSizeH 
            cellLbp = lbp[t:b, l:r]
            lbpHist.append(cal_hist(cellLbp, binSize).reshape(1, -1)[0])
    lbpHist = np.concatenate(lbpHist, dtype=np.float32)
    
    if normal:
        _range = np.max(lbpHist) - np.min(lbpHist)
        lbpHist = (lbpHist - np.min(lbpHist)) / _range
    return lbpHist

def cal_hist(gray, binSize=8):
    return cv2.calcHist([gray], [0], None, [binSize], [0, 256])


def extract_basic_lbp(image, flag=0):
    gray = cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),cv2.COLOR_BGR2GRAY)
    extractor = cal_lbp_lib if flag == 0 else cal_lbp_manual
    return extractor(gray)


def cal_lbp_lib(gray):
    lbp = skimage.feature.local_binary_pattern(gray,8,1.0,method='default')
    lbp = lbp.astype(np.uint8)
    return lbp


def cal_lbp_manual(gray):
    def cal_pixel_lbp(gray, x, y):
        h, w = gray.shape
        sum = 0
        # 0
        lx, ly = x-1, y-1
        if lx < 0 or ly < 0:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 1
        # 1
        lx, ly = x-1, y
        if lx < 0:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 2
        # 2
        lx, ly = x-1, y+1
        if lx < 0 or ly >= h:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 4
        # 3
        lx, ly = x, y+1
        if ly >= h:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 8
        # 4
        lx, ly = x+1, y+1
        if lx >= w or ly >= h:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 16
        # 5
        lx, ly = x+1, y
        if lx >= w:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 32
        # 6
        lx, ly = x+1, y-1
        if lx >= w or ly < 0:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 64
        # 7
        lx, ly = x, y-1
        if ly < 0:pass
        else: sum += 0 if gray[ly, lx] <= gray[y, x] else 128
        return sum
    
    h, w = gray.shape
    lap = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            lap[y, x] =  cal_pixel_lbp(gray, x, y)
    return lap


def show_lbp_result(rgb, lbp):
    plt.subplot(1,2,1)
    plt.imshow(rgb)
    plt.subplot(1,2,2)
    plt.imshow(lbp, cmap='gray')
    plt.show()


def diff_lbp_result(lbp1, name1, lbp2, name2):
    plt.subplot(1,2,1)
    plt.title(name1)
    plt.imshow(lbp1, cmap='gray')
    plt.subplot(1,2,2)
    plt.title(name2)
    plt.imshow(lbp2, cmap='gray')
    plt.show()


if __name__ == '__main__':
    image = cv2.imread('datasets/images/0000002.jpg')
    print(get_lbp_features(image, cellNums=(4, 4), binSize=64, flag=0, normal=True))


    
    


