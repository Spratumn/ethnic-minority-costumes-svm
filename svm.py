import os
import cv2
import random
import numpy as np
from sklearn import svm

import joblib
from tqdm import tqdm

from lbp import get_lbp_features
from eval import plot_confusion_matrix, plot_prediction
from pca import MyPCA

CELL_NUMS = (4, 8)
BIN_SIZE = 256
LBP_FLAG = 0
NORMAL = True
PCA_RATE = 0.1
USE_LIB_PCA = False


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


def split_dataset(dataList, train=0.7, test=0.3, shuffle=True):
    assert train + test == 1.0
    split = int(len(dataList) * train)
    if shuffle:
        random.seed(0)
        random.shuffle(dataList)
    
    trainSet = dataList[:split]
    testSet = dataList[split:]
    return trainSet, testSet

def load_sample(dataList):
    for (imagePath, id) in dataList:
        try:
            image = cv2.imread(imagePath)
            image.shape
            X = get_lbp_features(image, CELL_NUMS, BIN_SIZE, LBP_FLAG, NORMAL, PCA_RATE)
            y  = id
        except:
            pass
            # print('Invalid imagePath: ', imagePath)
        yield X, y


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


def run_train(dataDir, modelPath="./weights/svm_lpb_basic.weights"):
    train, _ = split_dataset(get_dataset(dataDir))
    trainX, trainy = load_data(train, dataDir + '/' + 'train')

    classifier = svm.SVC(C=1, kernel='rbf')
    if PCA_RATE != 1:
        assert 0 < PCA_RATE < 1
        pca = MyPCA(n_components=PCA_RATE, use_lib=USE_LIB_PCA)
        print('training pca with trainset...')
        pca.train(trainX)
        pca.save('./weights/pca.weights')
        trainX = pca.transform(trainX)
    
    classifier.fit(trainX, trainy)
    joblib.dump(classifier, modelPath)

    

def run_eval(dataDir, modelPath="./weights/svm_lpb_basic.weights"):
    classMap = {0: 'manzu', 1: 'mengguzu', 2: 'miaozu', 3: 'yaozu', 4: 'zhuangzu'}
    classLabels = ['manzu', 'mengguzu', 'miaozu', 'yaozu', 'zhuangzu']
    classifier = joblib.load(modelPath)
    train, test = split_dataset(get_dataset(dataDir))
    trainX, trainy = load_data(train, dataDir + '/' + 'train')
    testX, testy = load_data(test, dataDir + '/' + 'test')

    if PCA_RATE != 1:
        assert 0 < PCA_RATE < 1
        pca = MyPCA(n_components=PCA_RATE, use_lib=USE_LIB_PCA)
        pca.load('./weights/pca.weights')
        trainX = pca.transform(trainX)
        testX = pca.transform(testX)

    scoreTrain = classifier.score(trainX, trainy)
    scoreTest = classifier.score(testX,testy)
    print(f"The score of classifier is train: {scoreTrain}, test: {scoreTest}")

    trainPredDict = {0: ['manzu', 0, 0], 1: ['mengguzu', 0, 0], 2: ['miaozu', 0, 0], 3: ['yaozu', 0, 0], 4: ['zhuangzu', 0, 0]}
    trainPreds, trainGts = [], []
    for x, y in zip(trainX, trainy):
        trainPredDict[y][1] += 1
        pred = classifier.predict([x])[0]
        trainPreds.append(classMap[pred])
        trainGts.append(classMap[y])
        if pred == y:trainPredDict[y][2]+=1
    print(trainPredDict)
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
    run_train('./datasets')
    run_eval('./datasets')

    



