import os
import numpy as np
import pandas as pd

#import random 


from scipy.misc import imread
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import cv2
from imutils.perspective import four_point_transform
from imutils import contours
from skimage import io
from skimage.feature import hog
import imutils
from skimage.transform import resize, downscale_local_mean
import matplotlib.pyplot as plt
from keras.models import load_model


# Returns a compiled model identical to the previous one
model1 = load_model('G:/test/finalized_model_my_model.h5')







im = cv2.imread("C:/Users/suman/Pictures/Camera Roll/plate.jpg")
#image_resized = resize(im, (28,28))

#img = np.full((28,28,1), 12, dtype = np.uint8)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
image, ctrs, hier= cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[1],rect[0]), (rect[1] + rect[3],rect[0] + rect[2]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    #leng = int(rect[3]*1.12)
    #pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    #pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    
    roi = im_th[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    
    im1=pd.DataFrame(roi,dtype='float64')
    im2=im1.values.reshape(-1,28,28,1)
    im2-=np.mean(im2,axis=1)
    

    #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = model1.predict(im2)
    wer1=np.argmax(nbr)
    #nbr = np.reshape(nbr, (10,1))
    """
    nlabs=pd.DataFrame(nbr)
    
    num = nlabs[[0,1,2,3,4,5,6,7,8,9]].max(axis=1)

    for j in [0,1,2,3,4,5,6,7,8,9]:
       if(nlabs.ix[0][j]==num.ix[0]):
           wer1=j
           break
       else:
           continue
    """
    
    cv2.putText(im, str(wer1), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    print(wer1)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
