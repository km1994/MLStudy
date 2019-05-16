# coding:utf-8
# 一元线性回归分析例子

from sklearn import linear_model
import pandas as pd

#Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)
    times = []
    col1s = []
    col2s = []
    col3s = []
    col4s = []
    col5s = []
    col6s = []
    col7s = []
    col8s = []
    for time, col1,col2,col3,col4, col5,col6,col7,col8 in zip(data['time'], data['col1'],data['col2'],data['col3'],data['col4'],data['col5'],data['col6'],data['col7'],data['col8']):
        times.append([float(time)])
        col1s.append(float(col1))
        col2s.append(float(col2))
        col3s.append(float(col3))
        col4s.append(float(col4))
        col5s.append(float(col5))
        col6s.append(float(col6))
        col7s.append(float(col7))
        col8s.append(float(col8))
    return times,col1s,col2s,col3s,col4s,col5s,col6s,col7s,col8s,data

#Function for linear models
def linear_model_main(X_parameters, Y_parameters, predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_   #截距
    predictions['coefficient'] = regr.coef_   #回归系数
    predictions['predicted_value'] = predict
    return predictions

times,col1s,col2s,col3s,col4s,col5s,col6s,col7s,col8s,data = get_data('E:\\pythonWp\\pycharmWp\\algorithm\\DimensionalityReduction\\ARIMA\\bj.csv')


predict_time = 2017.4
result1 = linear_model_main(times, data['col1'], predict_time)
result2 = linear_model_main(times, data['col2'], predict_time)
result3 = linear_model_main(times, data['col3'], predict_time)
result4 = linear_model_main(times, data['col4'], predict_time)
result5 = linear_model_main(times, data['col5'], predict_time)
result6 = linear_model_main(times, data['col6'], predict_time)
result7 = linear_model_main(times, data['col7'], predict_time)
result8 = linear_model_main(times, data['col8'], predict_time)
times.append([float(2017.4)])
data['col1']['15']=float('%.2f' %result1['predicted_value'])
data['col2']['15']=float('%.2f' %result2['predicted_value'])
data['col3']['15']=float('%.2f' %result3['predicted_value'])
data['col4']['15']=float('%.2f' %result4['predicted_value'])
data['col5']['15']=float('%.2f' %result5['predicted_value'])
data['col6']['15']=float('%.2f' %result6['predicted_value'])
data['col7']['15']=float('%.2f' %result7['predicted_value'])
data['col8']['15']=float('%.2f' %result8['predicted_value'])

data=pd.DataFrame({'time':times,'col1':data['col1'],'col2':data['col2'],'col3':data['col3'],'col4':data['col4'],'col5':data['col5'],'col6':data['col6'],'col7':data['col7'],'col8':data['col8']})
print(data)
data.to_csv('E:\\pythonWp\\pycharmWp\\algorithm\\DimensionalityReduction\\ARIMA\\output.csv')
#pd.('E:\\pythonWp\\pycharmWp\\algorithm\\DimensionalityReduction\\ARIMA\\bj.csv')
