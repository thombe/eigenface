import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier('/Users/Thomas/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

# Indicate id number
id = 0

#names related to the index
names = ['None', 'Thomas', 'Thomas Smiling']

#Intialize and startup the video capture

cam = cv2.VideoCapture(0)
cam.set(3, 1280) # video width
cam.set(4, 720) # video heigth

#Define min window size to be recognizes as a face

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #check if confidence is less then 100 ===> 0 is pefect match
        if (confidence < 100):
            id = names[id]
            confidence = " {0}%".format(round(100-confidence))
        else:
            id = "Unknown"
            confidence = " {0}%".format(round(100-confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255) , 2)
        cv2.putText(img, str(confidence), (x + 5, y+h - 5), font, 1, (255, 255, 255), 1)


    cv2.imshow('camera' , img)
    k = cv2.waitKey(10) & 0xff # press 'ESC' to exit
    if k == 27:
        break

print('\n Exiting program and cleanup')
cam.release()
cv2.destroyAllWindows()
