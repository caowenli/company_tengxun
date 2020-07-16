# keras imports
from tensorflow.python.keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.backend import expand_dims
# std imports
import time
import gc
import os

from inputHandler import create_train_dev_set


class SiameseBiLSTM:
    def __init__(self, max_sequence_length, number_lstm, number_dense, rate_drop_lstm,
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio

    def train_model(self, sentences_pair, is_similar, model_save_directory='./'):
        train_data_1, train_data_2, labels_train, val_data_1, val_data_2, labels_val = create_train_dev_set(
            sentences_pair, is_similar, self.validation_split_ratio)
        if train_data_1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None
        # embedding_layer = Embedding(121, self.embedding_dim, input_length=self.max_sequence_length,
        #                             trainable=False)
        # Creating LSTM Encoder
        lstm_layer = Bidirectional(
            LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))
        # Creating LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='float32')
        sequence_1 = Lambda(lambda x: expand_dims(x, axis=-1))(sequence_1_input)
        # embedded_sequences_1 = embedding_layer(sequence_1_input)
        # x1 = lstm_layer(embedded_sequences_1)
        x1 = lstm_layer(sequence_1)
        # Creating LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='float32')
        sequence_2 = Lambda(lambda x: expand_dims(x, axis=-1))(sequence_2_input)

        # embedded_sequences_2 = embedding_layer(sequence_2_input)
        # x2 = lstm_layer(embedded_sequences_2)
        x2 = lstm_layer(sequence_2)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (
            self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)
        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        bst_model_path = checkpoint_dir + STAMP + '.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))
        model.fit([train_data_1, train_data_2], labels_train,
                  validation_data=([val_data_1, val_data_2], labels_val),
                  epochs=200, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])
        return bst_model_path
