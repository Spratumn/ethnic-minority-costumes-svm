import numpy as np
from numpy.linalg import eig
from sklearn.decomposition import PCA
import joblib

# 实现PCA处理
class MyPCA:
    def __init__(self, n_components=1, use_lib=0):
        # n_components：降维比例或降维后的特征维数
        # use_lib：是否使用库函数进行处理
        self.components = n_components
        self.principalEigenvectors = None
        self.use_lib = use_lib
    
    # 使用训练集数据进行pca训练
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
        mean = np.mean(X, axis=0)
        X -= mean
        convMatrix=np.dot(X.T, X)              # 计算X协方差矩阵
        eigenvalues,eigenvectors = eig(convMatrix)           # 计算协方差矩阵 特征值和特征向量
        eigPairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(X.shape[1])]
        eigPairs.sort(reverse=True)
        self.principalEigenvectors = np.array([ele[1] for ele in eigPairs[:componentNums]]).T
            
    # 对原始高维数据进行降维处理，得到低维数据
    def transform(self, X):
        if self.use_lib:
            return self._pca.transform(X)
        mean = np.mean(X, axis=0)
        return np.dot(X-mean, self.principalEigenvectors)
    
    # 保存pca训练参数
    def save(self, path):
        if self.use_lib:
            joblib.dump(self._pca, path)
        else:
            path = path.replace('.weights', '.npy')
            np.save(path, self.principalEigenvectors)
    
    # 加载pca训练参数
    def load(self, path):
        if self.use_lib:
            self._pca = joblib.load(path)
        else:
            path = path.replace('.weights', '.npy')
            self.principalEigenvectors = np.load(path)


if __name__ == '__main__':
    import cv2
    from lbp import get_lbp_features
    images = ['datasets/images/0000001.jpg',
              'datasets/images/0000002.jpg',
              'datasets/images/0000003.jpg',
              'datasets/images/0000004.jpg',
              'datasets/images/0000005.jpg',
              'datasets/images/0000006.jpg',
              'datasets/images/0000007.jpg',
              'datasets/images/0000008.jpg',]
    x = []
    for image in images:
        image = cv2.imread(image)
        x.append(get_lbp_features(image, cellNums=(4, 4), binSize=64, flag=0, normal=True))
    x = np.array(x)
    pca = MyPCA(n_components=2, use_lib=1)
    pca.train(x)
    print(pca.transform(x))
    pca_ = MyPCA(n_components=2, use_lib=0)
    pca_.train(x)
    print(pca_.transform(x))