# -*- coding: utf-8 -*-

"""
Created on Tue Apr 23 16:52:24 2019

@author: coda
"""

import tensorflow as tf
import nibabel as nib
import skimage.io as io
import numpy as np
import skimage.transform as trans
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'



    


def get_data(img_path,lab_path):
    
    print('-'*60)
    print('Loading images from ',img_path)
    print('Loading labels from ',lab_path)
    print('-'*60)

    # most of images are 880*880*12
    
    img_filename_list = os.listdir(img_path)
    
    id_list = []
    num = 0
    
    for i in img_filename_list:
    
        id_list.append(i.split('.')[0])
        img_filename = os.path.join(img_path, i)
        lab_filename = os.path.join(lab_path, ('mask_' + i.lower()) )
        
        img_nii = nib.load(img_filename)
        lab_nii = nib.load(lab_filename)
        img_arr = img_nii.get_fdata()
        lab_arr = lab_nii.get_fdata()
        
        #resize the images to 880*880*12
        img_re = trans.resize(img_arr,(880,880,12))
        lab_re = trans.resize(lab_arr,(880,880,12))
        
        # transpose the images
        img = np.transpose(img_re, [2, 0, 1])
        lab = np.transpose(lab_re, [2, 0, 1])
        
        if num == 0:
            train_img = img
            train_lab = lab
        else:
            train_img = np.concatenate((train_img, img), axis=0)
            train_lab = np.concatenate((train_lab, lab), axis=0)
        
        num += 1
        if num % 5 == 0:
            print('Loading data ...  ',num,'/',len(img_filename_list))
        
    train_img = train_img[...,np.newaxis]
    train_lab = train_lab[...,np.newaxis]

    print('The images and labels are all loaded.  The length is',len(img_filename_list))
    print('-'*60)
    
    return train_img,train_lab



def get_train_data():
    
    # train image and groungtruth should be prepared in 'SpineSegT2W/'
    
    img_path = "SpineSegT2W/image/"
    lab_path = "SpineSegT2W/groundtruth/"

    (train_img,train_lab) = get_data(img_path,lab_path)            

    np.save("data/train_img.npy",train_img)
    np.save("data/train_lab.npy",train_lab)
    
    return
    
    
    
def load_train_data():
    train_img = np.load("data/train_img.npy")
    train_lab = np.load("data/train_lab.npy")
    
    return train_img,train_lab