import pandas as pd

data = pd.read_csv(r'/Users/caowenli/Desktop/ml_pj/dl/time_series_similarity/generate_data.csv')
print(data.head(5))
print(data.keys())
columns = ['s_link_dir', 's_next_link_dir', 's_is_fork_road', 's_is_signal_light',
           's_link_dir_speed', 's_next_link_dir_speed']
print(data['s_is_fork_road'].value_counts())
print(data['s_is_signal_light'].value_counts())


# 定义正负样本
def defineLabel(df):
    res = []
    signal = df['s_is_signal_light'].tolist()
    fork = df['s_is_fork_road'].tolist()
    for i in range(len(signal)):
        if signal[i] == 1 and fork[i] == 1:
            res.append(0)
        elif signal[i] == 0 and fork[i] == 0 and res.count(1) < 500:
            res.append(1)
        else:
            res.append(2)
    return res


data['label'] = defineLabel(data)
print(data['label'].value_counts())

# 数据分割
train_data = data.loc[data['label'] != 2]
test_data = data.loc[data['label'] == 2]
