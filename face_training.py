import cv2
import numpy as np
from PIL import Image
import os

#path for face image database
path = 'dataset'

#recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(
    '/Users/Thomas/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

#function to get the images and label data
def getImgaesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    #For each picture
    facesPCA = []
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') #convert to gray
        img_numpy = np.array(PIL_img, 'uint8')

        #img_numpy is on the form [[255, 254, 255, 0,...,]\n [255,124,123,0,...,]]

        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids





print('Training faces. It will take a few seconds. Wait...')

faces,ids = getImgaesAndLabels(path)
#cv2.imshow('face' , faces[0])
#data = createDataMatrix(faces)
#recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml

#recognizer.write('trainer/trainer.yml')

#print the sum of faces trained and en program

print('\n {0} faces trained. Exiting program'.format(len(np.unique(ids))))