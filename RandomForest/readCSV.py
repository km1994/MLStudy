import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
data=pd.read_excel("xiechengscore.xlsx")
data = pd.DataFrame(data,columns=data.sheet_names)
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
data['species'] = pd.Factor(data.target, data.target_names)



#最小-最大值规范化  环境评分、服务评分、设施评分、用户推荐比、用户评分、评价内容
#酒店等级
data['hotelClass']=(data['hotelClass']-data['hotelClass'].min())/(data['hotelClass'].max()-data['hotelClass'].min())
#酒店最低价
data['hotelLowestprice']=(data['hotelLowestprice']-data['hotelLowestprice'].min())/(data['hotelLowestprice'].max()-data['hotelLowestprice'].min())
#酒店评论数
data['hotelComment']=(data['hotelComment']-data['hotelComment'].min())/(data['hotelComment'].max()-data['hotelComment'].min())
#用户推荐比
data['userRecommended']=(data['userRecommended']-data['userRecommended'].min())/(data['userRecommended'].max()-data['userRecommended'].min())
#卫生分数
data['healthScore']=(data['healthScore']-data['healthScore'].min())/(data['healthScore'].max()-data['healthScore'].min())
#环境评分
data['surroundingsScore']=(data['surroundingsScore']-data['surroundingsScore'].min())/(data['surroundingsScore'].max()-data['surroundingsScore'].min())
#服务评分
data['serviceScore']=(data['serviceScore']-data['serviceScore'].min())/(data['serviceScore'].max()-data['serviceScore'].min())
#设施评分
data['facilityScore']=(data['facilityScore']-data['facilityScore'].min())/(data['facilityScore'].max()-data['facilityScore'].min())

print(data)
features = data.columns[3:4]
clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(,data['surroundingsScore'],data['serviceScore'],data['facilityScore']], data['hotelComment'])

