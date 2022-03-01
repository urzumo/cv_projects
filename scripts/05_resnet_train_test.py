# CNN НА АРХИТЕКТУРЕ RESNET50 (NADAM)

# загрузим библиотеки
import numpy as np
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# функция выгрузки обучающего сета через загрузчик
def load_train(path):

    labels = pd.read_csv(path + 'labels.csv')
    train_datagen = ImageDataGenerator(
                          validation_split=0.25,
                          rescale=1./255,
                          horizontal_flip=True
                          )
    
    train_datagen_flow = (train_datagen.flow_from_dataframe(
                          dataframe=labels, 
                          directory=path + 'final_files/',
                          x_col='file_name', 
                          y_col='real_age', 
                          target_size=(150, 150), 
                          batch_size=8, 
                          class_mode='raw',
                          subset='training', 
                          seed=12345
                          )
    )

    return train_datagen_flow

# функция выгрузки тестового сета через загрузчик
def load_test(path):

    labels = pd.read_csv(path + 'labels.csv')
    test_datagen = ImageDataGenerator(
                          validation_split=0.25,
                          rescale=1./255
                          )
    test_datagen_flow = (test_datagen.flow_from_dataframe(
                          dataframe=labels, 
                          directory=path + 'final_files/',
                          x_col='file_name', 
                          y_col='real_age', 
                          target_size=(150, 150), 
                          batch_size=8, 
                          class_mode='raw',
                          subset='validation', 
                          seed=12345
                          )
    )

    return test_datagen_flow    
  
# функция создания модели
def create_model(input_shape):

    backbone = ResNet50(input_shape=input_shape,
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    optimizer=Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae'])
    
    return model

# функция обучения модели
def train_model(model, train_data, test_data, epochs=8, batch_size=None,
               steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data, 
              validation_data=(test_data), epochs=epochs,
              steps_per_epoch=steps_per_epoch, batch_size=batch_size,
              validation_steps=validation_steps, 
              verbose=2, shuffle=True)
    
    return model
