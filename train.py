import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, AvgPool2D, Dropout, ZeroPadding2D
from art.utils import load_dataset


def create_lenet_model(xshape):

    lenet = Sequential()
    lenet.add(ZeroPadding2D(padding=((2, 2), (2, 2)), input_shape=xshape))
    lenet.add(Conv2D(10, 5, activation='relu'))
    lenet.add(MaxPool2D(pool_size=(2, 2)))
    lenet.add(Conv2D(25, 5, activation='relu'))
    lenet.add(MaxPool2D(pool_size=(2, 2)))
    lenet.add(Conv2D(100, 4, activation='relu'))
    lenet.add(MaxPool2D(pool_size=(2, 2)))
    lenet.add(Flatten())
    lenet.add(Dense(10,activation='softmax'))
    return lenet


def create_cnn_model(xshape):

    cnn = Sequential()
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=xshape))
    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(MaxPool2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(128, (3, 3), activation='relu'))
    cnn.add(MaxPool2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(10, activation='softmax'))
    return cnn


def train_model(dataset):

    (x_train, y_train), (x_test, y_test), _, _ = load_dataset(str(dataset[0]))
    x_test = x_test * 2 - 1
    x_train = x_train * 2 - 1

    if dataset == ['mnist']:
        model = create_lenet_model(x_train.shape[1:])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))
        model.save_weights('./models/mnist.h5')
    elif dataset == ['cifar10']:
        model = create_cnn_model(x_train.shape[1:])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))
        model.save_weights('./models/cifar.h5')
    else:
        raise ValueError

    return model


