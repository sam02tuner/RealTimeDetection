import os
import numpy as np
import pandas as pd


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
#import random 


from scipy.misc import imread
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


train=pd.read_csv("G:/identify the digits/kaggle/train.csv", index_col=None)
test=pd.read_csv("G:/identify the digits/kaggle/test.csv", index_col=None)
sub=pd.read_csv("G:/identify the digits/kaggle/sample_submission.csv", index_col=None)

feature=train.pop("label")

train=train/255
test=test/255

temp=[]

train_x = train.values.reshape(-1,28,28,1)
test_y = test.values.reshape(-1,28,28,1)    
         
train_x-=np.mean(train_x,axis=0)
test_y-=np.mean(test_y,axis=0)


split_size=int(train_x.shape[0]*0.7)
train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = feature[:split_size], feature[split_size:]

train_y = keras.utils.to_categorical(train_y, 10)
val_y = keras.utils.to_categorical(val_y, 10)



def model():
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5),  activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    
    """
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    """

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
     
    return model

model1 = model()
epochs = 2
model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit(train_x, train_y,epochs=epochs, verbose=1, 
                   validation_data=(val_x, val_y))






import cv2
from imutils.perspective import four_point_transform
from imutils import contours
from skimage import io
from skimage.feature import hog
import imutils
from skimage.transform import resize, downscale_local_mean
import matplotlib.pyplot as plt








im = cv2.imread("C:/Users/suman/Pictures/Camera Roll/.png")
#image_resized = resize(im, (28,28))

#img = np.full((28,28,1), 12, dtype = np.uint8)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
image, ctrs, hier= cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3]*1.12)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    
    roi = im_th[pt1:pt1+leng, pt2+1:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    
    im1=pd.DataFrame(roi,dtype='float64')
    im2=im1.values.reshape(-1,28,28,1)
    im2-=np.mean(im2,axis=1)
    

    #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = model1.predict(im2)
    #nbr = np.reshape(nbr, (10,1))
    nlabs=pd.DataFrame(nbr)
    num = nlabs[[0,1,2,3,4,5,6,7,8,9]].max(axis=1)

    for j in [0,1,2,3,4,5,6,7,8,9]:
       if(nlabs.ix[0][j]==num.ix[0]):
           wer1=j
           break
       else:
           continue

    
    cv2.putText(im, str(wer1), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()

