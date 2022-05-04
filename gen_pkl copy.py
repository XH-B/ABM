#!/usr/bin/env python

import pickle as pkl

import numpy
  
from PIL import Image    
import cv2 
import os
    

paths= [your img dir ]
labels = [your label txt]


outFile = 'offline-train.pkl' 
outlabel= 'train_caption.txt'
oupFp_feature = open(outFile, 'wb')  
file_label = open(outlabel,'w')
features = {}
channels = 1
sentNum = 0



for image_path in paths:
    for i in os.listdir(image_path):
        print(i)
        key = str(i.split('.')[0])
        if os.path.exists(image_path + '/' + key + '.png' ):
            image_file = image_path + '/' + key + '.png' 
        else:
            image_file = image_path + '/' + key + '.bmp' 
        print(image_file)
        im = cv2.imread(image_file)  
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')  
        for channel in range(channels):
            mat[channel, :, :] = im[:,:,0] # 3 channel -> 1 channel
        sentNum = sentNum + 1
        features[key] = mat
        if sentNum % 500 == 0:
            print('process sentences ', sentNum)


for filename in labels:
    idx = 0  
    for line in open(filename):  
        file_label.writelines(line)  
        idx +=1
        
    
file_label.close()

print('load images done. sentence number ', sentNum)

pkl.dump(features, oupFp_feature)
print('save file done')
oupFp_feature.close()







