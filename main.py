# Written by Ashutosh Gupta
# Pyhton code to generate a funky frame over the human face using OpenCV with Haar Cascade approach.

import numpy as np
import cv2


#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

nose_cascade = cv2.CascadeClassifier('nose.xml')
shoulder_cascade = cv2.CascadeClassifier('shoulder.xml') 

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    upper_body = shoulder_cascade.detectMultiScale(gray, 1.3, 5)
    count = 0
    for (x,y,w,h) in upper_body:
        if count == 0:
            count += 1
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

            nose = 0
            for (a, b, c, d) in noses:
                #eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
                if nose == 0:
                    cv2.rectangle(roi_color,(a,b),(a+c,b+d),(0,255,0),2)
                    nose +=1
        
        # noses = nose_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in noses:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()