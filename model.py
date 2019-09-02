# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:47:32 2019

@author: coda
"""





import tensorflow.keras.models as models
from skimage.transform import resize
from skimage.io import imsave

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K





img_rows = 880
img_cols = 880
img_depth = 1
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def ppv_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return   (intersection) / (K.sum(y_pred_f))
    
def sen_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return   (intersection) / (K.sum(y_true_f))   


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def dpunet(pretrained_weights = None):
    
    inputs = Input((img_rows, img_cols, img_depth))
    
    # Block 1
    h1_11 = Convolution2D(16, 3, activation='relu', padding='same')(inputs)
    h1_12 = concatenate([inputs, h1_11], axis=3)
    h1_21 = Convolution2D(16, 3, activation='relu', padding='same')(h1_12)
    h1_22 = concatenate([inputs, h1_21], axis=3)
    p1 = MaxPooling2D(pool_size=(2, 2))(h1_22)

    # Block 2
    h2_11 = Convolution2D(32, 3, activation='relu', padding='same')(p1)
    h2_12 = concatenate([p1, h2_11], axis=3)
    h2_21 = Convolution2D(32, 3, activation='relu', padding='same')(h2_12)
    h2_22 = concatenate([p1, h2_21], axis=3)
    p2 = MaxPooling2D(pool_size=(2, 2))(h2_22)

    # Block 3
    h3_11 = Convolution2D(64, 3, activation='relu', padding='same')(p2)
    h3_12 = concatenate([p2, h3_11], axis=3)
    h3_21 = Convolution2D(64, 3, activation='relu', padding='same')(h3_12)
    h3_22 = concatenate([p2, h3_21], axis=3)
    p3 = MaxPooling2D(pool_size=(2, 2))(h3_22)

    # Block 4
    h4_11 = Convolution2D(128, 3, activation='relu', padding='same')(p3)
    h4_12 = concatenate([p3, h4_11], axis=3)
    h4_21 = Convolution2D(128, 3, activation='relu', padding='same')(h4_12)
    h4_22 = concatenate([p3, h4_21], axis=3)
    h4 = Dropout(0.5)(h4_22)
    p4 = MaxPooling2D(pool_size=(2, 2))(h4)

	
    # Block 5
	
    # hole = 2
    b1 = ZeroPadding2D(padding=(2, 2))(p4)
    b1 = Convolution2D(256, 3, 1, dilation_rate=(2, 2), activation='relu')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(256, 3, 1, activation='relu', padding='same')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(256, 3, 1, activation='relu', padding='same')(b1)

    # hole = 4
    b2 = ZeroPadding2D(padding=(4, 4))(p4)
    b2 = Convolution2D(256, 3, 1, dilation_rate=(4, 4), activation='relu')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(256, 3, 1, activation='relu', padding='same')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(256, 1, 1, activation='relu', padding='same')(b2)


    # hole = 6
    b3 = ZeroPadding2D(padding=(6, 6))(p4)
    b3 = Convolution2D(256, 3, 1, dilation_rate=(6, 6), activation='relu')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(256, 3, 1, activation='relu', padding='same')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(256, 3, 1, activation='relu', padding='same')(b3)

    # hole = 8
    b4 = ZeroPadding2D(padding=(8, 8))(p4)
    b4 = Convolution2D(256, 3, 1, dilation_rate=(8, 8), activation='relu')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(256, 3, 1, activation='relu', padding='same')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(256, 3, 1, activation='relu', padding='same')(b4)

	
	
    s = add([b1, b2, b3, b4])
    h5 = Conv2D(128, 3, activation = 'relu', padding = 'same')(s)
    h5 = Conv2D(128, 3, activation = 'relu', padding = 'same')(h5)

    u6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(h5), h4], axis=3)
    c6_11 = Conv2D(128, 3, activation = 'relu', padding = 'same')(u6)
    c6_12 = concatenate([u6, c6_11], axis=3)
    c6_21 = Conv2D(128, 3, activation = 'relu', padding = 'same')(c6_12)
    c6_22 = concatenate([u6, c6_21], axis=3)

    u7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6_22), h3_22], axis=3)
    c7_11 = Conv2D(64, 3, activation = 'relu', padding = 'same')(u7)
    c7_12 = concatenate([u7, c7_11], axis=3)
    c7_21 = Conv2D(64, 3, activation = 'relu', padding = 'same')(c7_12)
    c7_22 = concatenate([u7, c7_21], axis=3)
    
    
    u8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7_22), h2_22], axis=3)
    c8_11 = Conv2D(32, 3, activation = 'relu', padding = 'same')(u8)
    c8_12 = concatenate([u8, c8_11], axis=3)
    c8_21 = Conv2D(32, 3, activation = 'relu', padding = 'same')(c8_12)
    c8_22 = concatenate([u8, c8_21], axis=3)
    
    u9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8_22), h1_22], axis=3)
    c9_11 = Conv2D(16, 3, activation = 'relu', padding = 'same')(u9)
    c9_12 = concatenate([u9, c9_11], axis=3)
    c9_21 = Conv2D(16, 3, activation = 'relu', padding = 'same')(c9_12)
    c9_22 = concatenate([u9, c9_21], axis=3)

    c10 = Conv2D(1, (1, 1), activation='sigmoid')(c9_22)

    model = Model(inputs=[inputs], outputs=[c10])

    #model.summary()

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef,ppv_coef,sen_coef])


    if(pretrained_weights):
        model.load_weights(pretrained_weights)


    return model







