

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import tensorflow as tf
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import time
get_ipython().run_line_magic('matplotlib', 'inline')


def crop_image(s):
    image=Image.open(s)
    image.load()
    image_data = np.asarray(image)
    
    image_data_bw = image_data.max(axis=2)

    first_pixel = image_data_bw[0][0]

    non_empty_columns = np.where(image_data_bw.max(axis=0)!=first_pixel)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)!=first_pixel)[0]
    
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    
    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    
    new_image = Image.fromarray(image_data_new)
    return new_image, cropBox


def Detect(im):
#    im, cropBox = crop_image('Image5.JPG')
    im = np.asarray(im)
    im_orig = im
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
#     kernel = np.ones((2,2), np.uint8)
#     im = cv2.erode(im, kernel, iterations= 1)     
#     ret,thresh = cv2.threshold(im,127,255,cv2.THRESH_BINARY)

    thresh = cv2.Canny(im, 100, 200,apertureSize=5, L2gradient=True)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     for cnt in contours:
#         x,y,w,h = cv2.boundingRect(cnt)
#         if w>1 and h>1:
#             cv2.rectangle(im_orig,(x-1,y-1),(x+w+1,y+h+1),(255,0,0),1)
#             plt.imshow(im_orig)
#             f = plt.figure()
#             f.canvas.flush_events()
            
    i=0
    characters = []
    coords = []
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w>1 and h>1:
            characters.append(im_orig[y-1:y+h+1,x-2:x+w+1])
#            cv2.imwrite(str(i)+".jpg",im_orig[y-1:y+h+1,x-2:x+w+1], (cv2.IMWRITE_JPEG_QUALITY, 100))
            i=i+1
            coords.append([x-2,y-1,w+3,h+2])
        
    return characters, coords



def pad(c_temp):
    if(c_temp.shape[0] < 28 and c_temp.shape[1] < 28):
        c_temp = cv2.copyMakeBorder(c_temp, 
                                    0,
                                    abs(28-c_temp.shape[0]),
                                    0,
                                    abs(28-c_temp.shape[1]),
                                    borderType=cv2.BORDER_REPLICATE)
        
    elif(c_temp.shape[0] > 28 and c_temp.shape[1] < 28):
        c_temp = cv2.copyMakeBorder(c_temp, 
                                    0,
                                    0,
                                    0,
                                    abs(28-c_temp.shape[1]),
                                    borderType=cv2.BORDER_REPLICATE)
        c_temp = c_temp[:28, :]
        
    elif(c_temp.shape[0] < 28 and c_temp.shape[1] > 28):
        c_temp = cv2.copyMakeBorder(c_temp, 
                                    0,
                                    abs(28-c_temp.shape[0]),
                                    0,
                                    0,
                                    borderType=cv2.BORDER_REPLICATE)
        c_temp = c_temp[:, :28]
        
    else:
        c_temp = c_temp[:28, :28]
        
    return c_temp


def pad(c_temp):
    if(c_temp.shape[0] < 28 and c_temp.shape[1] < 28):
        if c_temp.shape[0]%2==0 and c_temp.shape[1]%2==0:
            c_temp = cv2.copyMakeBorder(c_temp, 
                                        abs(28-c_temp.shape[0])//2,
                                        abs(28-c_temp.shape[0])//2,
                                        abs(28-c_temp.shape[1])//2,
                                        abs(28-c_temp.shape[1])//2,
                                        borderType=cv2.BORDER_REPLICATE)
        elif c_temp.shape[0]%2==0 and c_temp.shape[1]%2!=0:
            c_temp = cv2.copyMakeBorder(c_temp, 
                                        abs(28-c_temp.shape[0])//2,
                                        abs(28-c_temp.shape[0])//2,
                                        abs(28-c_temp.shape[1])//2+1,
                                        abs(28-c_temp.shape[1])//2,
                                        borderType=cv2.BORDER_REPLICATE)
        elif c_temp.shape[0]%2!=0 and c_temp.shape[1]%2==0:
            c_temp = cv2.copyMakeBorder(c_temp, 
                                        abs(28-c_temp.shape[0])//2+1,
                                        abs(28-c_temp.shape[0])//2,
                                        abs(28-c_temp.shape[1])//2,
                                        abs(28-c_temp.shape[1])//2,
                                        borderType=cv2.BORDER_REPLICATE)
        else:
            c_temp = cv2.copyMakeBorder(c_temp, 
                                        abs(28-c_temp.shape[0])//2+1,
                                        abs(28-c_temp.shape[0])//2,
                                        abs(28-c_temp.shape[1])//2+1,
                                        abs(28-c_temp.shape[1])//2,
                                        borderType=cv2.BORDER_REPLICATE)
            
    elif(c_temp.shape[0] > 28 and c_temp.shape[1] < 28):
        c_temp = cv2.copyMakeBorder(c_temp, 
                                    0,
                                    0,
                                    0,
                                    abs(28-c_temp.shape[1]),
                                    borderType=cv2.BORDER_REPLICATE)
        c_temp = c_temp[:28, :]
        
    elif(c_temp.shape[0] < 28 and c_temp.shape[1] > 28):
        c_temp = cv2.copyMakeBorder(c_temp, 
                                    0,
                                    abs(28-c_temp.shape[0]),
                                    0,
                                    0,
                                    borderType=cv2.BORDER_REPLICATE)
        c_temp = c_temp[:, :28]
        
    else:
        c_temp = c_temp[:28, :28]
        
    return c_temp



def withModel(s):
    im_orig = Image.open(s)
    im, cropBox = crop_image(s)
    chars, coords = Detect(im)
    model = models.load_model('synthetic_dataset4.h5')
    
    classes = '0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
    new_chars = []
    
    for c in chars:
        
        c_temp = pad(c)
        c_temp = cv2.cvtColor(c_temp, cv2.COLOR_BGR2GRAY)
        c_temp = c_temp.astype('float32')/255       
        plt.imshow(c_temp)
        f = plt.figure()            
        f.canvas.flush_events()
        
        c_temp = c_temp.reshape(1, 28, 28, 1)
        
        predictions = model.predict(c_temp, batch_size=1, verbose=0)[0]
        
        print(classes[np.argmax(predictions)])
        
        if(classes[np.argmax(predictions)] in '0 1 2 3 4 5 6 7 8 9'.split()):
            (s1, s2, s3) = c.shape
            c = c.tolist()
            for i in range(s1):
                for j in range(s2):
                    for k in range(s3):
                        c[i][j][k] = 0
            c = np.array(c)
            
        new_chars.append(c)
    
    for i in range(len(new_chars)):
        new_chars[i] = Image.fromarray(new_chars[i].astype('uint8'), 'RGB')
        
    for i in range(len(new_chars)):
        im.paste(new_chars[i], (coords[i][0], coords[i][1]))
        
    im_orig = np.asarray(im_orig)
    im = np.asarray(im)
    
    (s1, s2, s3) = im.shape
    im_orig = im_orig.tolist()
    for i in range(s1):
        for j in range(s2):
            for k in range(s3):
                im_orig[i + cropBox[0]][j + cropBox[2]][k] = im[i][j][k]
    im_orig = np.array(im_orig)
    
    Image.fromarray(im_orig.astype(np.uint8)).save('Result_'+s.split('.')[0]+'.JPG')

    plt.imshow(im_orig)



withModel('Image6.jpg')





