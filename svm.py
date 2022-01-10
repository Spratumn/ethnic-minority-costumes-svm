import os
import cv2
import random
import numpy as np
from sklearn import svm
import time
import joblib
from tqdm import tqdm

from lbp import get_lbp_features
from eval import plot_confusion_matrix, plot_prediction
from pca import MyPCA
from svm_config import *

# 加载数据集（图像数据路径及标签列表）
def get_dataset(datasetDir):
    annoPath = os.path.join(datasetDir, 'annotations.csv')
    assert os.path.exists(annoPath)
    with open(annoPath, 'r') as annoF:
        lines = annoF.readlines()
    annoList = []
    for line in lines:
        imagePath, id = line.rstrip('\n').split(',')
        imagePath = datasetDir + '/' + imagePath
        annoList.append((imagePath, int(id)))
    return annoList

# 划分数据集，将上面方法加载的图像数据路径及标签列表按照比例划分为训练集和测试集
def split_dataset(dataList, train=0.7, test=0.3, shuffle=True):
    assert train + test == 1.0
    split = int(len(dataList) * train)
    if shuffle:
        random.seed(0)
        random.shuffle(dataList)
    
    trainSet = dataList[:split]
    testSet = dataList[split:]
    return trainSet, testSet

# 逐个读取图像数据并提取lbp特征
def load_sample(dataList):
    
    for (imagePath, id) in dataList:
        try:
            image = cv2.imread(imagePath)
            image.shape
            X = get_lbp_features(image, CELL_NUMS, BIN_SIZE, LBP_LIB_FLAG, NORMAL)
            y  = id
            yield X, y
        except InterruptedError:
            break
        except AttributeError:
            pass
            # print('Invalid imagePath: ', imagePath)
        

# 从已有本地文件中直接读取特征数据或重新从图像创建lbp特征数据（加快训练流程免得反复进行特征提取过程）
def load_data(dataList, cachePath=''):
    Xpath = f'{cachePath}_x.npy'
    ypath = f'{cachePath}_y.npy'
    try:
        Xs, ys = np.load(Xpath), np.load(ypath)
        print('loaded cache data from ', cachePath)
    except:
        print('creating cache data to ', cachePath)
        Xs, ys = [], []
        for X, y in tqdm(load_sample(dataList)):
            Xs.append(X)
            ys.append(y)
        Xs, ys = np.array(Xs), np.array(ys)
        np.save(Xpath, Xs)
        np.save(ypath, ys)
    return Xs, ys

# svm训练过程
def run_train(dataDir, modelPath="./weights/svm_lpb_basic.weights"):
    # 1.加载并划分训练集和验证集
    train, _ = split_dataset(get_dataset(dataDir))
    
    # 2.加载训练集特征数据
    trainX, trainy = load_data(train, dataDir + '/' + 'train')
    
    # 3.创建svm分类器
    classifier = svm.SVC(C=1, kernel='rbf')

    # 4.根据配置参数进行pca降维处理
    if PCA_RATE > 0:
        pca = MyPCA(n_components=PCA_RATE, use_lib=PCA_LIB_FLAG)
        print('training pca with trainset...')
        pca.train(trainX)
        pca.save('./weights/pca.weights')
        trainX = pca.transform(trainX)
    # 5.使用训练数据对分类器进行训练
    print('training SVM with trainset...')
    classifier.fit(trainX, trainy)
    # 6.保存分类器训练权重
    joblib.dump(classifier, modelPath)

    
# svm测试过程
def run_eval(dataDir, modelPath="./weights/svm_lpb_basic.weights"):
    # 1.设置类别id映射关系
    classMap = {0: 'manzu', 1: 'mengguzu', 2: 'miaozu', 3: 'yaozu', 4: 'zhuangzu'}
    classLabels = ['manzu', 'mengguzu', 'miaozu', 'yaozu', 'zhuangzu']
    # 2. 加载分类器训练权重
    classifier = joblib.load(modelPath)
    # 3. 加载测试集数据
    train, test = split_dataset(get_dataset(dataDir))
    trainX, trainy = load_data(train, dataDir + '/' + 'train')
    testX, testy = load_data(test, dataDir + '/' + 'test')
    # 4.根据配置参数进行pca降维处理
    if PCA_RATE > 0:
        pca = MyPCA(n_components=PCA_RATE, use_lib=PCA_LIB_FLAG)
        pca.load('./weights/pca.weights')
        trainX = pca.transform(trainX)
        testX = pca.transform(testX)
    # 5.使用训练集和测试集数据进行分类器评分
    print('cal score with trainset...')
    scoreTrain = classifier.score(trainX, trainy)
    print(f"The score of trainset: {scoreTrain}")
    print('cal score with testset...')
    scoreTest = classifier.score(testX,testy)
    print(f"The score of testset: {scoreTest}")
    
    # 6.使用分类器生成预测结果并绘制混淆矩阵和预测结果柱状图
    trainPredDict = {0: ['manzu', 0, 0], 1: ['mengguzu', 0, 0], 2: ['miaozu', 0, 0], 3: ['yaozu', 0, 0], 4: ['zhuangzu', 0, 0]}
    trainPreds, trainGts = [], []
    start = time.time()
    for x, y in zip(trainX, trainy):
        trainPredDict[y][1] += 1
        pred = classifier.predict([x])[0]
        trainPreds.append(classMap[pred])
        trainGts.append(classMap[y])
        if pred == y:trainPredDict[y][2]+=1
    print(f'mean predict time cost: {(time.time() - start) / len(trainy)}')
    plot_prediction(trainPredDict)
    plot_confusion_matrix(trainPreds, trainGts, classLabels)

    testPredDict = {0: ['manzu', 0, 0], 1: ['mengguzu', 0, 0], 2: ['miaozu', 0, 0], 3: ['yaozu', 0, 0], 4: ['zhuangzu', 0, 0]}
    testPreds, testGts = [], []
    for x, y in zip(testX, testy):
        testPredDict[y][1] += 1
        pred = classifier.predict([x])[0]
        testPreds.append(classMap[pred])
        testGts.append(classMap[y])
        if classifier.predict([x])[0] == y:testPredDict[y][2]+=1
    plot_prediction(testPredDict)
    plot_confusion_matrix(testPreds, testGts, classLabels)
        


if __name__ == '__main__':
    dataDir = './datasets'
    
    start = time.time()
    run_train(dataDir)
    print(f'train time cost: {time.time() - start}')
    
    run_eval(dataDir)
    



