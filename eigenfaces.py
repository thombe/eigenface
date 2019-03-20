import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

#path for face image database
path = 'dataset'

imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
ids = np.empty(len(imagePaths) , dtype=int)

X_t = np.zeros((len(imagePaths),10000) , dtype='f')

count = 0
#For each picture
for imagePath in imagePaths:

    PIL_img = Image.open(imagePath).convert('L') #convert to gray
    img_numpy = np.array(PIL_img, 'uint8')

    X_t[count] = img_numpy.ravel()

    id = int(os.path.split(imagePath)[-1].split(".")[1])

    ids[count] = id
    count += 1



h = 100
w = 100

target_names = ['Thomas' , 'ThomasSmile' , 'ThomasSad' , 'Driverlicense']

# split into a training and testing set.
# test_size represents how many of og data used for testing.
X_train, X_test, y_train, y_test = train_test_split(X_t, ids, test_size=0.0)


# Compute a PCA with n_components and whitened for better preformance
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# apply PCA transformation to training data
X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)


# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade = cv2.CascadeClassifier('/Users/Thomas/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(3, 1280) # video width
cam.set(4, 720) # video heigth

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
        #Creates rectangle around face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        #Extracts the face from grayimg, resizes and flattens
        face = gray[y:y+h,x:x+w]
        face = cv2.resize(face, (100,100))
        face = face.ravel()

        #Computes the PCA of the face
        faceTestPCA = pca.transform(face.reshape(1, -1))
        pred = clf.predict(faceTestPCA)

        print(target_names[pred[0]])

        cv2.putText(img, str(target_names[pred[0]]), (x+5,y-5), font, 1, (255,255,255) , 2)

    cv2.imshow('camera' , img)
    k = cv2.waitKey(10) & 0xff # press 'ESC' to exit
    if k == 27:
        break

print('\n Exiting program and cleanup')
cam.release()
cv2.destroyAllWindows()

'''
print(classification_report(y_test, y_pred, target_names=target_names))


# Visualization
def plot_gallery(images, titles, h, w, rows=2, cols=2):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)


eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface {0}".format(i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
'''