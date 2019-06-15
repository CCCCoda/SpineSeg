# -*- coding: utf-8 -*-

"""
Created on Tue Apr 23 16:52:24 2019

@author: coda
"""


from model import *
from data import *
from predict import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



load_train_data()

model = briunet(pretrained_weights = 'model_bri.h5')
model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
model.fit(train_img, train_lab, batch_size=6, epochs=10,callbacks=[model_checkpoint],validation_split=0.2) #epoch30

#model.save('model_bri_01.h5')

predict_test_images()