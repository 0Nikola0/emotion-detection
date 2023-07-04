import os

import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


def create_train_test_datasets(data, num_labels=7):
    train_x, train_y, test_x, test_y = [], [], [], []
    for i, row in data:
        val = row['pixels'].split(" ")
        try:
            if 'Training' in row['Usage']:
                train_x.append(np.array(val,'float32'))
                train_y.append(row['emotion'])
            elif 'PublicTest' in row['Usage']:
                test_x.append(np.array(val,'float32'))
                test_y.append(row['emotion'])
        except:
            print(f"error occured at index :{i} and row:{row}")
    
    # Convert to float32 np array
    train_x = np.array(train_x,'float32')
    train_y = np.array(train_y,'float32')
    test_x = np.array(test_x,'float32')
    test_y = np.array(test_y,'float32')

    # Change labels to categorical
    train_y = np_utils.to_categorical(train_y, num_classes=num_labels)
    test_y = np_utils.to_categorical(test_y, num_classes=num_labels)

    # Normalize data
    train_x -= np.mean(train_x, axis=0)
    train_x /= np.std(train_x, axis=0)

    test_x -= np.mean(test_x, axis=0)
    test_x /= np.std(test_x, axis=0)

    # Change data dimension
    train_x = train_x.reshape(train_x.shape[0], 48, 48, 1)
    test_x = test_x.reshape(test_x.shape[0], 48, 48, 1)
    
    return train_x, train_y, test_x, test_y


def create_model(train_x, num_labels=7):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(train_x.shape[1:])))
    model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 3rd Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(Flatten())

    # Fully Connected layer 1st layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    # Fully Connected layer 2nd layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_labels, activation='softmax'))

    # Compile model
    model.compile(loss=categorical_crossentropy,
                    optimizer=Adam(),
                    metrics=['accuracy'])

    return model


def export_model(model, json_out="cnn_model.json", h5_out="cnn_model_weights.h5"):
    # === Export model === #
    model_json = model.to_json()

    with open(json_out, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(h5_out)


def load_model(json_out="cnn_model.json", h5_out="cnn_model_weights.h5"):
    # load model architecture
    with open(json_out, "r") as file:
        json_savedModel = file.read()

    model = keras.models.model_from_json(json_savedModel)
    model.load_weights(h5_out)
                       
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='SGD',
                    metrics=['accuracy'])

    return model


def main():
    BASE_DIR = os.getcwd()
    df = pd.read_csv("fer2013.csv")
    data = df.iterrows()

    train_x, train_y, test_x, test_y = create_train_test_datasets(data)

    model = create_model(train_x)
    # === Training === #
    model.fit(train_x, train_y,
            batch_size=64,
            epochs=30,
            # TODO verbose=1 za DEBUG (pokazuva nekoi printovi)
            verbose=0,
            validation_data=(test_x, test_y),
            shuffle=True)
    # === End Training === #

    export_model(model)


if __name__ == "__main__":
    main()
