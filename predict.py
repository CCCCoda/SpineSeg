# -*- coding: utf-8 -*-

"""
Created on Tue May 14 17:35:21 2019

@author: coda
"""

import tensorflow as tf
import nibabel as nib
import skimage.io as io
import numpy as np
import skimage.transform as trans

from model import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def predict_nii(test_path,result_path):

    
    
    test_filename_list = os.listdir(test_path)
    id_list = []
    
    #num = 0
    
    print('-'*60)
    print('Predicting',len(test_filename_list),'images from ',test_path)
    print('-'*60)
    print('Loading model...')
    print('-'*60)
    
    model = briunet(pretrained_weights = 'model_bri.h5')
    
    for i in test_filename_list:
    
        id_list.append(i.split('.')[0])
        img_filename = os.path.join(test_path, i)
        pre_filename = os.path.join(result_path, ('predict_' + i.lower()) )
        
        # load image
        img_nii = nib.load(img_filename)
        img_arr = img_nii.get_fdata()
        
        #lab_nii = nib.load(lab_filename) baocunlujing
        # get the shape of image
        img_rows = img_arr.shape[0]
        img_cols = img_arr.shape[1]
        img_depth = img_arr.shape[2]
        
        img_re = trans.resize(img_arr,(880,880,img_depth))
        img = np.transpose(img_re, [2, 0, 1])
        test_img = img[...,np.newaxis]
        
        print('-'*60)
        print('Predicting image ',i)
        print('-'*60)
        
        results = model.predict(test_img, verbose=1)
        predict_img = results[:,:,:,0]
        predict_img = np.transpose(predict_img, [1, 2, 0])
        
        # resize to initial size of images
        predict_img = trans.resize(predict_img,(img_rows,img_cols,img_depth))
        # 0.5
        
        # Save predicted images
        predict_nii = nib.nifti1.Nifti1Image(predict_img,None,header=img_nii.header)
        nib.save(predict_nii,pre_filename)
        
    return


def predict_test_images():
    
    test_path = "test/image/"
    result_path = "test/result/"
    
    predict_nii(test_path,result_path)         
    return
