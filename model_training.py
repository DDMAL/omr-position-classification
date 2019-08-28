from keras.layers import Input, Concatenate,concatenate, Dense, Embedding, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D, Activation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from sklearn.utils import class_weight
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
import cv2 as cv
import time
import math
import tensorflow as tf
import pandas as pd

def train_model(img_path, coords, positions, avg_neume_height, output_model_path):

    data_im = []
    data_position = []

    img = cv.imread(img_path)

    for i, c in enumerate(coords):
        neume_img = img[
        c[0]-2*avg_neume_height:c[0]+c[2]+2*avg_neume_height,
        c[1]:c[1]+c[3]]
        neume_img = cv.resize(neume_img, (30,120), interpolation=cv.INTER_AREA)
        neume_img = neume_img / 255.0
        data_im.append(neume_img)
        data_position.append(positions[i])

    data = pd.DataFrame(
        {'image': data_im,
         'position': data_position,
        })

    position_classes = len(data.groupby('position'))

    df_data = data.groupby('position').filter(lambda x : len(x)>4)

    position_weights = class_weight.compute_class_weight('balanced',
      np.unique(df_data['position']),
      df_data['position'])

    lrr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    early_stopping_min = EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=0, mode='auto')

    data_position = df_data['position']
    data_im = [(image / 1) for image in df_data['image']]
    data_im = np.asarray(data_im).reshape(len(data_im), 120,30,3)
    data_position = np.asarray(data_position)

    position_encoder = LabelEncoder()
    position_encoder.fit(data_position)

    position_categories = []

    for pos in data_position:
      if pos not in position_categories:
        position_categories.append(pos)

    position_categories.sort()

    im_train, im_test, position_train, position_test = train_test_split(
        data_im, data_position,
        test_size=0.25,
        stratify=data_position,
        random_state=2
    )

    label_encoder = LabelEncoder()

    label_encoder.fit(data_position)
    position_train_nn = label_encoder.transform(position_train)
    position_train_nn = to_categorical(position_train_nn, num_classes=position_classes)
    position_test_nn = label_encoder.transform(position_test)
    position_test_nn = to_categorical(position_test_nn, num_classes=position_classes)

    n_folds = 5
    epochs = 15
    batch = 32

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    position_checkpoint = ModelCheckpoint(output_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    for train, test in kfold.split(data_im, data_position):

      X_train_im = data_im[train]
      X_test_im = data_im[test]

      X_train_pos = data_position[train]
      X_test_pos = data_position[test]


      Y_train_pos_nn = position_encoder.transform(df_data['position'].iloc[train])
      Y_train_pos_nn = to_categorical(Y_train_pos_nn, num_classes=position_classes)

      Y_test_pos_nn = position_encoder.transform(df_data['position'].iloc[test])
      Y_test_pos_nn = to_categorical(Y_test_pos_nn, num_classes=position_classes)

      model_position = position_model(position_classes)

      model_position.fit(
        X_train_im, Y_train_pos_nn,
        batch_size = batch, epochs = epochs,
        validation_data = (X_test_im,Y_test_pos_nn),
        callbacks=[lrr,position_checkpoint,early_stopping],verbose=2,
        class_weight=position_weights)

    return position_checkpoint

def position_model(categories,optimizer='rmsprop'):
  kernel_size = (3,3)
  filters = 32
  dropout = 0.25
  pool_size = (2,2)
  inputs = Input(shape=(120, 30, 3))
  y = Conv2D(filters=32, kernel_size=kernel_size,activation='relu',padding='same')(inputs)
  y = Conv2D(filters=32, kernel_size=kernel_size,activation='relu',padding='same')(y)
  y = MaxPooling2D(pool_size=pool_size)(y)
  y = Dropout(dropout)(y)

  y = Conv2D(filters=64,kernel_size=kernel_size,activation='relu',padding='same')(y)
  y = Conv2D(filters=64,kernel_size=kernel_size,activation='relu',padding='same')(y)
  y = MaxPooling2D(pool_size=pool_size,strides=(2,2))(y)
  y = Dropout(dropout)(y)
  y = Flatten()(y)
  # dropout regularization
  y = Dense(256,activation='relu')(y)
  y = Dropout(dropout)(y)
  outputs = Dense(categories, activation='softmax')(y)
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  return model
