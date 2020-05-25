import cv2
import os
from keras.models import load_model
import numpy as np
import time
from identify import label_identification



upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('sample_video2.mp4')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
frame_no = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame = cap.read()
height,width = frame.shape[:2]
out = cv2.VideoWriter('new_sample_output.mp4',fourcc, 20.0, (width,height))
print(height,width)
while(True):
    ret, img = cap.read()
    frame_no += 1
    print(frame_no)
    if ret ==True:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
          roi_gray = gray[y:y+h+2,x:x+w+2]
          roi_color = img[y:y+h+2,x:x+w+2]
          label = label_identification(roi_color)
          if label == 'mask':
            cv2.putText(img,'with mask',(int(x-3),int(y-3)),font,int((y/x)+1),(23,200,76),2,cv2.LINE_AA)
            cv2.rectangle(img,(x,y),(x+w,y+h),(23,200,76),3)
          elif label == 'no_mask':
            cv2.putText(img,'without mask',(int(x-3),int(y-3)),font,int((y/x)+1),(0,0,255),2,cv2.LINE_AA)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        out.write(img)
            
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
print('Detection is Done')