# АУГМЕНТАЦИЯ CNN НА АРХИТЕКТУРЕ LENET (ADAM)

# загрузим библиотеки
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# функция выгрузки обучающего сета через загрузчик
def load_train(path):
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=42
    )

    return train_datagen_flow
  
# функция создания модели
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=5, activation='relu',
                                  padding='same', input_shape=input_shape))
    model.add(AvgPool2D(pool_size=2, strides=2))
    
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1), padding='valid', 
                    activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
    
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1),
                    activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))

    optimizer = Adam(lr=1e-3) 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    return model
  
# функция обучения модели
def train_model(model, train_data, test_data, batch_size=None, epochs=10,
                steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model
