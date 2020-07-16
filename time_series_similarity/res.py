import pandas as pd
from sklearn.metrics import accuracy_score

train_data = pd.read_csv(r'/Users/caowenli/Desktop/ml_pj/dl/time_series_similarity/train_data.csv')
print(train_data.keys())
data = train_data[['s_link_dir', 's_next_link_dir', 'label', 'predict']]
data.to_csv('train_data_valid.csv', index=None)
print('acc', accuracy_score(train_data['label'], train_data['predict']))
test_data = pd.read_csv(r'/Users/caowenli/Desktop/ml_pj/dl/time_series_similarity/test_data.csv')
test_data = test_data[:500]
test_data = test_data[['s_link_dir', 's_next_link_dir', 'predict']]
test_data.to_csv('test_data_valid.csv', index=None)
