import matplotlib.pyplot as plt
from skimage import data, exposure
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
import numpy as np
import cv2


### data preparing
cars_amount = 1000
cats_amount = 1000
data_amount = 2000

# cars : target=0
cars_images = []
for x in range(cars_amount):
    img = cv2.imread('./cars/%05d.jpg' % (x+1))
    img = cv2.resize(img, (300, 200))
    cars_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cars_images = np.array(cars_images)
cars_target = [0 for i in range(0, cars_amount)]
cars_target = np.array(cars_target)

# cats : target=1
cats_images = []
for x in range(cats_amount):
    img = cv2.imread('./cats/cat.%d.jpg' % (x))
    img = cv2.resize(img, (300, 200))
    cats_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cats_images = np.array(cats_images)
cats_target = [1 for i in range(0, cats_amount)]
cats_target = np.array(cats_target)

# merge two sets
images = list(cars_images) + list(cats_images)
target = list(cars_target) + list(cats_target)
images = np.array(images)
target = np.array(target)
# 20 pictures for prediction

prediction_images = []
# cars
for x in range(10):
    img = cv2.imread('./prediction/%05d.jpg' % (x+1001))
    img = cv2.resize(img, (300, 200))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# cats
for x in range(10):
    img = cv2.imread('./prediction/cat.%d.jpg' % (x+1990))
    img = cv2.resize(img, (300, 200))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

prediction_images = np.array(prediction_images)
prediction_target = [0]*10 + [1]*10
prediction_target = np.array(prediction_target)

### SIFT

des_list = []
sift = cv2.SIFT_create()

for img in images:
    kpts = sift.detect(img)
    kpts, des = sift.compute(img, kpts)
    des_list.append(des)

descriptors = des_list[0]
for descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

### 20 K-means
k = 20
voc, variance = kmeans(descriptors, k, 1)

### generate features
# 對於訓練集中的每一張圖片，統計vocabulary中K個word的“詞頻”，得到相應的直方圖
im_features = np.zeros((data_amount, k), 'float32')
for i in range(data_amount):
    words, distance = vq(des_list[i], voc)
    for w in words:
        im_features[i][w] += 1

# Standardization 平均&變異數標準化
# stdSlr = StandardScaler().fit(im_features)
# im_features = stdSlr.transform(im_features)

x_train, x_test, y_train, y_test = train_test_split(
                        im_features , target,
                        test_size=0.25, random_state=0)


### svm
# clf = svm.SVC(C=1, kernel='linear', gamma='scale', dual=True, max_iter = 1000)
clf = LinearSVC(dual=True, max_iter = 3000)
clf.fit(x_train, y_train)

### accuracy
print("Train Data Accuracy")
print(clf.score(x_train, y_train))
print("Test Data Accuracy")
print(clf.score(x_test, y_test))

### 20 Prediction test

prediction_des_list = []
for img in prediction_images:
    kpts = sift.detect(img)
    kpts, des = sift.compute(img, kpts)
    prediction_des_list.append(des)

prediction_descriptors = prediction_des_list[0]
for descriptor in prediction_des_list[1:]:
    descriptors = np.vstack((prediction_descriptors, descriptor))

### 20 K-means
k = 20
voc, variance = kmeans(descriptors, k, 1)

### generate features
prediction_im_features = np.zeros((20, k), 'float32')
for i in range(20):
    words, distance = vq(prediction_des_list[i], voc)
    for w in words:
        prediction_im_features[i][w] += 1

# stdSlr = StandardScaler().fit(prediction_im_features)
# prediction_im_features = stdSlr.transform(prediction_im_features)

print("Predictin Accuracy")
print(clf.score(prediction_im_features, prediction_target))