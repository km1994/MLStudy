from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(df)
#print("----------------")
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
#print(df)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(iris.target)
print(iris.target_names)
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]
#print(train)
#print("-----")
#print(test)
features = df.columns[:4]
#print(features)
clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['species'])
clf.fit(train[features], y)

preds = iris.target_names[clf.predict(test[features])]


#print(pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds']))
