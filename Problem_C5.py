# ============================================================================================
# PROBLEM C5
#
# Build and train a neural network model using the Daily Min Temperature.csv dataset.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://github.com/jbrownlee/Datasets
#
# Desired MAE < 0.19 on the normalized dataset.
# by Hedrin S. Sitorus
# ============================================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import urllib
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import SGD


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_C5():
    data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    urllib.request.urlretrieve(data_url, 'daily-min-temperatures.csv')

    time_step = []
    temps = []

    with open('daily-min-temperatures.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            temps.append(np.float32(row[1]))# YOUR CODE HERE)
            time_step.append(step) # YOUR CODE HERE)
            step=step + 1

    series= np.array(temps)

    # Normalization Function. DO NOT CHANGE THIS CODE
    min=np.min(series)
    max=np.max(series)
    series -= min
    series /= max
    time=np.array(time_step)

    # DO NOT CHANGE THIS CODE
    split_time=2500

    time_train= time[:split_time]
    x_train= series[:split_time]
    time_valid= time[split_time:]
    x_valid= series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size=64
    batch_size=256
    shuffle_buffer_size=1000

    train_set=windowed_dataset(
        x_train, window_size, batch_size, shuffle_buffer_size)
    print(train_set)
    print(x_train.shape)

    valid_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)

    model=tf.keras.models.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    # YOUR CODE HERE
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    model.fit(train_set, epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.0001)], validation_data=valid_set)

    # Get the model's prediction on the test set
    test_pred = model.predict(valid_set).flatten()

    # Only use the first 500 samples of the test set
    test_pred = test_pred[:len(x_valid)]
    x_valid = x_valid[:len(x_valid)]

    # Plot the predictions
    plt.plot(test_pred, 'r-', label='Prediction')
    plt.plot(x_valid, 'b-', label='Truth')

    plt.legend()
    plt.show()

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_C5()
    model.save("model_C5.h5")
