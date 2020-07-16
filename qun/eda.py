import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
data = pd.read_csv(r'/Users/caowenli/Desktop/ml_pj/dl/qun/data_qun.csv')
data.keys()
features = ['国家', '年份', '自变量', '城镇人口（占总人口比例）', '人均 GDP（2010年不变价美元）',
            '外国直接投资额（以当前价格计算，百万美元）', '第一产业占比']
data = data[features]
print(data.describe())
print(data.info())
data.dropna(inplace=True, subset=['自变量', '城镇人口（占总人口比例）', '人均 GDP（2010年不变价美元）',
                                  '外国直接投资额（以当前价格计算，百万美元）', '第一产业占比'])


import math


def log_function(value):
    if value == 0:
        return value
    flag = True if value > 0 else False
    value = abs(value)
    return math.log(value) if flag else -math.log(value)


log_features = ['自变量', '城镇人口（占总人口比例）', '人均 GDP（2010年不变价美元）',
                '外国直接投资额（以当前价格计算，百万美元）', '第一产业占比']
for i in log_features:
    data["log_" + i] = data[i].apply(lambda x: log_function(x))

data.to_excel("data_all.xlsx", index=None)
describe = data.describe()
describe.to_excel("describe.xlsx")