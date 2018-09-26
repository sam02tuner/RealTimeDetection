import numpy as np
import cv2
import pandas as pd
import os


from keras.models import load_model


# Returns a compiled model identical to the previous one
model = load_model('G:/test/finalized_model_my_model.h5')


#from imutils import contours




"""
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #cv2.imshow('frame',frame)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""



cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    x, y, w, h = 0, 0, 300, 300
    ret, im = cap.read()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow('im',im_gray)
    
    blur = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
    ret, thresh1 = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    thresh1 = thresh1[y:y + h, x:x + w]
# Find contours in the image
    img, contours, hier= cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                rects = [cv2.boundingRect(ctr) for ctr in contours]
                for rect in rects:
                # Draw the rectangles
                # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
                    newImage = thresh1[rect[1]:rect[1] + rect[3], rect[0]:rect[0]+ rect[2]]
                    newImage = cv2.resize(newImage, (28, 28))
                    newImage = np.array(newImage)
                    im1=pd.DataFrame(newImage,dtype='float64')
                    im2=im1.values.reshape(-1,28,28,1)
                    im2-=np.mean(im2,axis=1)
                    nbr = model.predict(im2)
                    nb = np.argmax(nbr)

                    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
                    cv2.putText(im, str(nb), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
                    #cv2.putText(im, str(nbr), (5,300),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    

    #cv2.putText(img, "Deep Network :  " + str(1), (10, 380),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh1)
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    k = cv2.waitKey(10)
    if k == 27:
       break

   
    
    
    
cap.release()
cv2.destroyAllWindows()   
    
    
    
    
    