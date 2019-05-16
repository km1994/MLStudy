import numpy as np
import pandas as pd

#导入数据集
data_train = pd.read_csv('xiechengNoZerosTrain.csv', header=None)
data_test = pd.read_csv('xiechengNoZerosTest.csv',  header=None)
print("data_train:",data_train)
print("data_test:",data_test)
train_x,train_y = data_train[np.arange(7)],data_train[7]
print("train_x:",train_x)
print("train_y:",train_y)
test_x,test_y = data_test[np.arange(7)],data_test[7]
print("test_x:",test_x)
print("test_y:",test_y)
from sklearn.decomposition import PCA
estimator = PCA(n_components=7)
pca_train_x = estimator.fit_transform(train_x)
pca_train_x=pd.DataFrame(pca_train_x)
print("pca_train_x:",pca_train_x)
pca_train_x.to_csv("pca_train_x.csv")

pca_test_x = estimator.transform(test_x)
pca_test_x=pd.DataFrame(pca_test_x)
print("pca_test_x:",pca_test_x)
pca_test_x.to_csv("pca_test_x.csv")
