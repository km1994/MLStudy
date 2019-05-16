import numpy as np
import pandas as pd
#导入数据集
digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)
print(digits_train)
print(digits_test)
#分割数据与标签
train_x,train_y = digits_train[np.arange(64)],digits_train[64]
test_x,test_y = digits_test[np.arange(64)],digits_test[64]

print("train_x:",train_x)

#主成分分析
from sklearn.decomposition import PCA
estimator = PCA(n_components=20)
pca_train_x = estimator.fit_transform(train_x)
print("pca_train_x:",pca_train_x)
pca_test_x = estimator.transform(test_x)
print("pca_test_x:",pca_test_x)

#训练支持向量机
from sklearn.svm import LinearSVC
#原始数据
print("原始数据")
svc = LinearSVC()
svc.fit(X=train_x,y=train_y)
print(svc.score(test_x,test_y))

#PCA处理后数据
print("PCA处理后数据")
svc_pca = LinearSVC()
svc_pca.fit(pca_train_x,train_y)
print(svc_pca.score(pca_test_x,test_y))
