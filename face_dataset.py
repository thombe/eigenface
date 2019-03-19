import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) # video width
cam.set(4, 360) # video heigth

face_detector = cv2.CascadeClassifier(
    '/Users/Thomas/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

#For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ===>  ')
print("\n Initializing face capture. Look at the camera and wait...")

# Initialize individual sampling face count
count = 0

while (True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the datasets folder
        grayFace = gray[y:y+h,x:x+w]
        grayFace = cv2.resize(grayFace, (100,100))
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + '.jpg',
                    grayFace)
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 5:# take 5 face sample and stop video
        break

#Do a bit of cleanup
print("\n Exiting program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


