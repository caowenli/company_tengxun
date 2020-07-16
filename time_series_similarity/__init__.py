import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing import sequence

text1 = '1 2 5 6 9'
text2 = '3 4 7 8 10 11 12 12 23'
texts = [text1, text2]

tokenizer = Tokenizer(num_words=None)  # num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
tokenizer.fit_on_texts(texts)
print(tokenizer.word_counts)  # [('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)]
print(tokenizer.word_index)  # {'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
print(tokenizer.word_docs)  # {'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
print(tokenizer.index_docs)  # {1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
# num_words=多少会影响下面的结果，行数=num_words
tmp = tokenizer.texts_to_sequences(texts)  # 得到词索引[[1, 2, 3, 4], [1, 2, 3, 5]]
print(tokenizer.texts_to_matrix(texts))  # 矩阵化=one_hot
test_data_1 = pad_sequences(tmp, maxlen=5)
print(test_data_1)
print(type(test_data_1))