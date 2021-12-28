import numpy as np
from numpy.linalg import eig
from sklearn.decomposition import PCA
import joblib


class MyPCA:
    def __init__(self, n_components=1, use_lib=False):
        self.components = n_components
        self.principalEigenvectors = None
        self.use_lib = use_lib

    def train(self, X):
        if self.components < 1:
            oriLength = X.shape[1]
            componentNums = int(self.components * oriLength)
        else:
            componentNums = self.components
        if self.use_lib:
            self._pca = PCA(n_components=componentNums)
            self._pca.fit(X)
            return

        X = X - np.mean(X,axis=0)
        covMat = np.cov(X.T, ddof = 0)                           # 计算X协方差矩阵
        eigenvalues,eigenvectors = eig(np.mat(covMat))           # 计算协方差矩阵 特征值和特征向量
        indexes = np.argsort(eigenvalues)                      # 选取最大的K个特征值及其特征向量

        principalIndexes=indexes[-1:-(componentNums+1):-1]   # 最大的n个特征值的下标
        self.principalEigenvectors = eigenvectors[:, principalIndexes]      

    def transform(self, X):
        if self.use_lib:
            return self._pca.transform(X)
        X = X - np.mean(X,axis=0)
        X = np.dot(X, self.principalEigenvectors)            # 用X与特征向量相乘
        return X                   
    
    def save(self, path):
        if self.use_lib:
            joblib.dump(self._pca, path)
        else:
            np.save(path, self.principalEigenvectors)

    def load(self, path):
        if self.use_lib:
            self._pca = joblib.load(path)
        else:
            self.principalEigenvectors = np.load(path)


