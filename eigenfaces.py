import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
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


# Load data
#lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

h = 100
w = 100

#_, h, w = lfw_dataset.images.shape
#X = lfw_dataset.data
#y = lfw_dataset.target
target_names = ['Thomas' , 'ThomasSmile' , 'EddieSmile' , 'Eddie']

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_t, ids, test_size=0.3)


# Compute a PCA 
n_components = 7
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# apply PCA transformation to training data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)


y_pred = clf.predict(X_test_pca)
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
