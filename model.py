from __future__ import print_function

import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import load_model
from sklearn.preprocessing import normalize

NUM_CLASSES = 6


# Load data to X and y
def load_data_for_fc(features_path, label_path, label_index, train_ratio):
    X_data = pd.read_csv(features_path, header=None)
    X_data.drop(X_data.columns[0], axis=1, inplace=True)

    y_data = pd.read_csv(label_path, header=None)
    y_data = y_data[label_index]

    data = pd.concat([X_data, y_data], axis=1)
    data = data.sample(frac=1)
    size = len(data)
    train_size = int(train_ratio * size)

    X_train = data.iloc[0:train_size, 0:X_data.shape[1]]
    X_test = data.iloc[train_size: size, 0:X_data.shape[1]]
    y_train = data.iloc[0:train_size, X_data.shape[1]:X_data.shape[1] + 1]
    y_test = data.iloc[train_size: size, X_data.shape[1]:X_data.shape[1] + 1]

    X_train = X_train.as_matrix()
    y_train = y_train.as_matrix()
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix()

    X_train = normalize(X_train, axis=1, norm='l1')
    X_test = normalize(X_test, axis=1, norm='l1')

    return X_train, y_train, X_test, y_test


def load_data_for_cnn(features_path, label_path, label_index, train_ratio, hist_rows, hist_cols):
    X_train, y_train, X_test, y_test = load_data_for_fc(features_path, label_path, label_index, train_ratio)
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, hist_rows, hist_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, hist_rows, hist_cols)
        # input_shape = (1, hist_rows, hist_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], hist_rows, hist_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], hist_rows, hist_cols, 1)
        # input_shape = (hist_rows, hist_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return X_train, y_train, X_test, y_test


def build_fc_model(input_size, hidden_layers, hidden_units, output_size):
    layers = []
    layers.append(keras.layers.Dense(hidden_units, activation=tf.nn.relu, input_shape=(input_size,)))
    for i in range(0, hidden_layers):
        layers.append(keras.layers.Dense(hidden_units, activation=tf.nn.relu))
    layers.append(keras.layers.Dense(output_size, activation=tf.nn.softmax))
    model = keras.Sequential(layers=layers)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_model(input_shape, hidden_layers, hidden_units, output_size):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    for i in range(0, hidden_layers):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(output_size, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train(model_type, hidden_layers, hidden_units, X_train, y_train, X_test, y_test, input_shape):
    EPOCHS = 100

    if model_type == 'fc':
        model = build_fc_model(X_train.shape[1], hidden_layers, hidden_units, NUM_CLASSES)
    else:
        model = build_cnn_model(input_shape, hidden_layers, hidden_units, NUM_CLASSES)

    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(filepath='models/model_checkpoint.h5', monitor='val_loss', save_best_only=True)]

    start_time = time.time()
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=callbacks)
    model.save('models/best_model.h5')
    model.save_weights('models/best_model_weights.h5')
    elapsed_time = time.time() - start_time
    train_acc = np.array(history.history['acc'])[-1]
    test_loss, test_acc = model.evaluate(X_test, y_test)
    return train_acc, test_acc, elapsed_time, model.count_params()


def get_cnn_input_shape(hist_rows, hist_cols):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, hist_rows, hist_cols)
    else:
        input_shape = (hist_rows, hist_cols, 1)
    return input_shape


def predict_using_saved_model(model_type, hidden_layers, hidden_units, X_test, hist_rows, hist_cols):
    input_shape = get_cnn_input_shape(hist_rows, hist_cols)
    if model_type == 'fc':
        model = build_fc_model(X_test.shape[1], hidden_layers, hidden_units, NUM_CLASSES)
    else:
        model = build_cnn_model(input_shape, hidden_layers, hidden_units, NUM_CLASSES)
    model = load_model('models/best_model.h5')
    model.load_weights('models/best_model_weights.h5')
    return model.predict_classes(X_test)


def main():

    model_types = ['fc', 'cnn']
    # model_type = model_types[0]

    # hist_sizes = ["10x10", "30x30", "50x50", "100x100"]
    units = [2, 4, 6, 8, 10]
    # layers = [1, 2, 3, 4, 5]

    hist_sizes = ['50x50']
    # units = [10]
    layers = [3]

    output_f = open('algorithm_selection_varying_units.csv', 'w')
    for hist_size in hist_sizes:
        hist_rows, hist_cols = int(hist_size.split('x')[0]), int(hist_size.split('x')[1])
        features_path = 'data/train_and_test/hist_{0}.csv'.format(hist_size)
        label_path = 'data/train_and_test/label_{0}.csv'.format(hist_size)

        input_shape = get_cnn_input_shape(hist_rows, hist_cols)

        for model_type in model_types:
            if model_type == 'fc':
                X_train, y_train, X_test, y_test = load_data_for_fc(features_path, label_path, 1, 0.8)
            else:
                X_train, y_train, X_test, y_test = load_data_for_cnn(features_path, label_path, 1, 0.8, hist_rows,
                                                                     hist_cols)

            for num_layers in layers:
                for num_units in units:
                    train_acc, test_acc, elapsed_time, params = train(model_type, num_layers, num_units, X_train,
                                                                      y_train,
                                                                      X_test, y_test, input_shape)
                    print("{0},{1},{2},{3},{4},{5},{6},{7}".format(hist_size, model_type, num_layers, num_units, train_acc, test_acc,
                                                               elapsed_time, params))
                    output_f.writelines("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(hist_size, model_type, num_layers, num_units, train_acc, test_acc,
                                                               elapsed_time, params))

    output_f.close()


if __name__ == '__main__':
    main()
