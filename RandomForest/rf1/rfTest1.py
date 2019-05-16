import pandas as pd
#1. 读入数据
#从本地读入“wheat.csv”文件，指定index_col参数为00，即将第一列作为每行的索引。用head()函数查看前几行数据。
data = pd.read_csv("wheat.csv",index_col=0)
print(data.head(6))

#2. 缺失值处理
#该数据集中包含部分缺失值，在模型训练时会遇到特征值为空的问题，故对缺失值进行处理，
## 用DataFrame的fillna方法进行缺失值填充，填充值为用mean方法得到的该列平均值。
data = data.fillna(data.mean())
print(data)


#3. 划分数据集从sklearn.model_selection模块导入train_test_split函数，
# 并将返回值放入变量X_train、X_test、y_train和y_test之中，指定参数test_size=0.3，
# 即将70%的数据样本作为训练集，将30%的数据样本作为测试集。输出训练集和测试集大小。
from sklearn.model_selection import train_test_split
X = data.iloc[:,:7]
y = data.iloc[:,7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=300)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#4.构建随机森林模型并训练
#从sklearn.ensemble模块中导入RandomForestClassifier函数，
## 并用其构建随机森林分类模型，指定n_estimators参数为1010，
# 即使用1010棵决策树构建模型。将训练集传入模型进行模型训练。
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

#5.利用随机森林模型预测分类
#运用predict方法预测测试集中样本的分类，该方法返回一个预测结果数组，输出预测的分类结果。
y_pred = model.predict(X_test)
print("Predictions of test set:\n%s"%y_pred)

#6. 查看各特征重要性
#用feature_importances_属性查看每个特征的重要性，相对而言第11、22、55、77个特征在随机森林分类中的重要性强一些。
print(model.feature_importances_)

#7. 评估模型准确率
#利用score方法计算模型在测试集上的预测准确率。
print(model.score(X_test,y_test))

#8. 调整随机森林中的树木数量
#随机森林中的数目数量是模型中较为重要的参数，
#通过指定n_estimators参数进行设置，设置为30时模型的性能较10时有所提升，
#但设置为100时，其准确度不但没有提升已不明显，甚至可能下降，可能已经过拟合。
model= RandomForestClassifier(n_estimators=30)
model.fit(X_train, y_train)
print(model.score(X_test,y_test))

#9. 与决策树分类进行比较
#决策树与随机森林在分类效果上进行比较，
# 决策树模型的分类准确率与仅含单棵决策树的随机森林类似，
# 但是总体上随机森林的准确度要高于决策树，但其模型的解释性较差，无法轻易得到树的基本结构。
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))

model= RandomForestClassifier(n_estimators=1)
model.fit(X_train, y_train)
print(model.score(X_test,y_test))