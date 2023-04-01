import cv2 
import numpy as np 

cv2.namedWindow("test") 
cap=cv2.VideoCapture(0) 
success, frame = cap.read() 
color = (0,0,0) 
classfier=cv2.CascadeClassifier("opencv\data\haarcascades\haarcascade_frontalface_alt.xml") 
while success: 
    success, frame = cap.read() 
    size=frame.shape[:2] 
    image=np.zeros(size,dtype=np.float16) 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    cv2.equalizeHist(image, image) 
    divisor=8 
    h, w = size 
    minSize =(w//divisor, h//divisor) 
    faceRects = classfier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,minSize) 
    if len(faceRects)>0: 
        for faceRect in faceRects: 
            x, y, w, h = faceRect 
            cv2.rectangle(frame, (x, y), (x+w, y+h), color) 
    cv2.imshow("test", frame) 
    key=cv2.waitKey(10) 
    c = chr(key & 255) 
    if c in ['q', 'Q', chr(27)]: 
        break 
cv2.destroyWindow("test")