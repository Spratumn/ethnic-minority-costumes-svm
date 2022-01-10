import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import skimage.feature
import skimage.segmentation
import time
from tqdm import tqdm


# 从给定图像提取基于lbp的直方图特征
def get_lbp_features(image, cellNums=(4, 4), binSize=64, flag=0, normal=False):
    """
    cellNums: LBP划分区域数量（WN, HN）
    binSize: 直方图分箱数量
    flag: 是否使用sklearn库提供的接口进行LBP提取
    normal: 是否对直方图特征进行归一化处理
    """
    # 1.提取整幅图像的lbp特征
    lbp = extract_basic_lbp(image=image, flag=flag)
    
    # 2. 将lbp特征按照cellNums划分子区域进行直方图统计
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
    # 3.将子区域的直方图串接起来
    lbpHist = np.concatenate(lbpHist, dtype=np.float32)
    
    # 4. 对直方图特征进行归一化处理
    if normal:
        _range = np.max(lbpHist) - np.min(lbpHist)
        lbpHist = (lbpHist - np.min(lbpHist)) / _range
    
    return lbpHist


def cal_hist(gray, binSize=8):
    return cv2.calcHist([gray], [0], None, [binSize], [0, 256])

# 对给定图像提取lbp特征
def extract_basic_lbp(image, flag=0):
    gray = cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),cv2.COLOR_RGB2GRAY)
    extractor = cal_lbp_lib if flag == 1 else cal_lbp_manual
    return extractor(gray)

# 使用sklearn提供的lbp特征提取方法
def cal_lbp_lib(gray):
    lbp = skimage.feature.local_binary_pattern(gray,8,1.0,method='default')
    lbp = lbp.astype(np.uint8)
    return lbp

# 自己手写实现的lbp特征提取方法
def cal_lbp_manual(gray):
    h, w = gray.shape
    lbp = np.zeros((h, w))
    # 逐行逐列对图像进行lbp特征计算
    for y in range(h):
        for x in range(w):
            lbp[y, x] =  cal_pixel_lbp(gray, x, y)
    
    lbp = lbp.astype(np.uint8)
    return 255 - lbp

# 对图像中坐标（x,y）处的像素位置计算lbp特征值
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

# 绘制lbp特征图像
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


# 对lbp特征提取进行时间统计
def test_time_cost(datasetDir, cellNums=(4, 8), binSize=256, normal=True):
    imageDir = datasetDir + '/images'
    imagenames = os.listdir(imageDir)
    start = time.time()
    total = 0
    for imagename in tqdm(imagenames):
        image = cv2.imread(os.path.join(imageDir, imagename))
        try:
            image.shape
            get_lbp_features(image, cellNums=cellNums, binSize=binSize, flag=1, normal=normal)
            total += 1
        except InterruptedError:
            break
        except AttributeError:
            pass
    print(f'config: cellNums=({cellNums}), binSize={binSize}, normal={normal}')
    print(f'mean time cost is {(time.time() - start) / total}')

if __name__ == '__main__':
    datasetDir = './datasets'
    image = cv2.imread('datasets/images/0000001.jpg')
    lbp = extract_basic_lbp(image, flag=1)
    show_lbp_result(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), lbp)
    # print(get_lbp_features(image, cellNums=(4, 8), binSize=256, flag=0, normal=True))

    # test_time_cost(datasetDir, cellNums=(6, 12), binSize=256, normal=True)



    
    


