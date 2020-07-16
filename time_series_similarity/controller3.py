from model import SiameseBiLSTM
from inputHandler import create_test_data
from config import siamese_config
import numpy as np
from operator import itemgetter
from tensorflow.python.keras.models import load_model
import pandas as pd
import gc

########################################
############ Data Preperation ##########
########################################

# print(data.head(5))
# print(data.keys())
# columns = ['s_link_dir', 's_next_link_dir', 's_is_fork_road', 's_is_signal_light',
#            's_link_dir_speed', 's_next_link_dir_speed']
# print(data['s_is_fork_road'].value_counts())
# print(data['s_is_signal_light'].value_counts())
#

data = pd.read_csv(r'/Users/caowenli/Desktop/ml_pj/dl/time_series_similarity/generate_data_all_2.csv')


# 定义正负样本
def defineLabel(df):
    res = []
    signal = df['s_is_signal_light'].tolist()
    fork = df['s_is_fork_road'].tolist()
    for i in range(len(signal)):
        if signal[i] == 1 and fork[i] == 1:
            res.append(0)
        elif signal[i] == 0 and fork[i] == 0 and res.count(1) < 1000:
            res.append(1)
        else:
            res.append(2)
    return res


data['label'] = defineLabel(data)
print(data['label'].value_counts())

# 数据分割
train_data = data.loc[data['label'] != 2]
test_data = data.loc[data['label'] == 2]

sentences1 = list(train_data['s_link_dir_speed'])
sentences2 = list(train_data['s_next_link_dir_speed'])
is_similar = list(train_data['label'])
print(sentences1[0])
print(sentences2[0])
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]

#########################
####### Training ########
##########################
#
# from config import siamese_config
#
#
# class Configuration(object):
#     """Dump stuff here"""
#
#
# CONFIG = Configuration()
#
# CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
# CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
# CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
# CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
# CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
# CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
# CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']
#
# siamese = SiameseBiLSTM(CONFIG.max_sequence_length, CONFIG.number_lstm_units,
#                         CONFIG.number_dense_units,
#                         CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function,
#                         CONFIG.validation_split_ratio)
#
# best_model_path = siamese.train_model(sentences_pair, is_similar, model_save_directory='./')

########################
###### Testing #########
########################
# #
model = load_model(
    r'/Users/caowenli/Desktop/ml_pj/dl/time_series_similarity/checkpoints/1594292125/lstm_128_64_0.30_0.30.h5')
# model = load_model(best_model_path)
train_data_x1, train_data_x2 = create_test_data(sentences_pair)
train_preds = list(model.predict([train_data_x1, train_data_x1], verbose=1).ravel())
train_res = []
for i in train_preds:
    if i <= 0.5:
        train_res.append(0)
    else:
        train_res.append(1)
print(len(train_res))
from sklearn.metrics import confusion_matrix

print(train_res)
train_data['predict'] = train_res
print(confusion_matrix(train_data['predict'].tolist(), train_data['label'].tolist()))
train_data.to_csv("train_data_7_9_deep.csv", index=None)
print(train_data['predict'].value_counts())

test_sentences1 = list(test_data['s_link_dir_speed'])
test_sentences2 = list(test_data['s_next_link_dir_speed'])
test_sentences_pair = [(x1, x2) for x1, x2 in zip(test_sentences1, test_sentences2)]
test_data_x1, test_data_x2 = create_test_data(test_sentences_pair)
preds = list(model.predict([test_data_x1, test_data_x2], verbose=1).ravel())
test_res = []
for i in preds:
    if i < 0.5:
        test_res.append(0)
    else:
        test_res.append(1)
test_data['predict'] = test_res
print(confusion_matrix(test_data['predict'].tolist(), test_data['label'].tolist()))
test_data.to_csv("test_data_7_9_deep.csv", index=None)
print(test_data['predict'].value_counts())

data_sentences1 = list(data['s_link_dir_speed'])
data_sentences2 = list(data['s_next_link_dir_speed'])
data_sentences_pair = [(x1, x2) for x1, x2 in zip(data_sentences1, data_sentences2)]
data_data_x1, data_data_x2 = create_test_data(data_sentences_pair)
preds = list(model.predict([data_data_x1, data_data_x2], verbose=1).ravel())
data_res = []
for i in preds:
    if i < 0.5:
        data_res.append(0)
    else:
        data_res.append(1)
data['predict'] = data_res
print(confusion_matrix(data['predict'].tolist(), data['label'].tolist()))
data.to_csv("data_data_7_9_deep.csv", index=None)
print(data['predict'].value_counts())
