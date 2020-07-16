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

data = pd.read_csv(r'/Users/caowenli/Desktop/ml_pj/dl/time_series_similarity/select_data_labeled.csv')
print(data.keys())
data['true_label'].fillna(1, inplace=True)
data['true_label'] = data['true_label'].astype(dtype=int)
print(data['true_label'].value_counts())
print(data.info())


# 定义正负样本
def defineLabel(df):
    res = []
    signal = df['s_is_signal_light'].tolist()
    fork = df['s_is_fork_road'].tolist()
    for i in range(len(signal)):
        if signal[i] == 1 and fork[i] == 1:
            res.append(0)
        else:
            res.append(1)
    return res


data['label'] = defineLabel(data)
print(data['label'].value_counts())
# 数据分割
sentences1 = list(data['s_link_dir_speed'])
sentences2 = list(data['s_next_link_dir_speed'])
is_similar = list(data['true_label'])
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]

########################
###### Training ########
#########################

from config import siamese_config


class Configuration(object):
    """Dump stuff here"""


CONFIG = Configuration()
CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, CONFIG.number_lstm_units,
                        CONFIG.number_dense_units,
                        CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function,
                        CONFIG.validation_split_ratio)

best_model_path = siamese.train_model(sentences_pair, is_similar, model_save_directory='./')

#######################
##### Testing #########
#######################
#
# model = load_model(
#     r'/Users/caowenli/Desktop/ml_pj/dl/time_series_similarity/checkpoints/1593418708/lstm_128_64_0.17_0.30.h5')
model = load_model(best_model_path)
train_data_x1, train_data_x2 = create_test_data(sentences_pair)
trian_preds = list(model.predict([train_data_x1, train_data_x1], verbose=1).ravel())
res = []
for i in trian_preds:
    if i < 0.5:
        res.append(0)
    else:
        res.append(1)
print(len(res))
print(res)
data['predict'] = np.array(res)
data.to_csv("labeled_train_data.csv", index=None)

test_data = pd.read_csv('test_data.csv')
test_sentences1 = list(test_data['s_link_dir_speed'])
test_sentences2 = list(test_data['s_next_link_dir_speed'])
test_sentences_pair = [(x1, x2) for x1, x2 in zip(test_sentences1, test_sentences2)]
test_data_x1, test_data_x2 = create_test_data(test_sentences_pair)
preds = list(model.predict([test_data_x1, test_data_x2], verbose=1).ravel())
# results = [(x, y, z) for (x, y), z in zip(test_sentences_pair, preds)]
# results.sort(key=itemgetter(2), reverse=True)
# print(results)
test_res = []
for i in preds:
    if i < 0.5:
        test_res.append(0)
    else:
        test_res.append(1)
print(len(test_res))
test_data['predict'] = test_res
test_data.to_csv("test_data.csv", index=None)
