from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import gc


def create_train_dev_set(sentences_pair, is_similar, validation_split_ratio):
    sentences1 = [x[0] for x in sentences_pair]
    sentences1 = [x.split() for x in sentences1]
    sentences2 = [x[1] for x in sentences_pair]
    sentences2 = [x.split() for x in sentences2]
    train_padded_data_1 = np.array(sentences1, dtype=int)
    train_padded_data_2 = np.array(sentences2, dtype=int)
    train_labels = np.array(is_similar, dtype=int)
    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, val_data_1, val_data_2, labels_val


def create_test_data(test_sentences_pair):
    """
    Create training and validation dataset
    Args:
        test_sentences_pair (list): list of tuple of sentences pairs
    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    sentences1 = [x[0] for x in test_sentences_pair]
    sentences1 = [x.split() for x in sentences1]
    sentences2 = [x[1] for x in test_sentences_pair]
    sentences2 = [x.split() for x in sentences2]
    return np.array(sentences1, dtype=int), np.array(sentences2, dtype=int)
