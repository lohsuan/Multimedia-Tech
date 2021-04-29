import matplotlib.pyplot as plt
from skimage import data, exposure
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
import numpy as np
import cv2


### data preparing
paper_amount = 300
sicer_amount = 291
stone_amount = 300
data_amount = 891

# paper_amount = 100
# sicer_amount = 100
# stone_amount = 100
# data_amount = 300

# paper_amount = 219
# sicer_amount = 195
# stone_amount = 173
# data_amount = 587

# paper : target=0
paper_images = []
for x in range(paper_amount):
    img = cv2.imread('./paper/paper_%03d.jpg' % (x))
    # img = cv2.imread('./paper_000/paper_%03d.jpg' % (x))
    img = cv2.resize(img, (300, 300))
    paper_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# paper_images = np.array(paper_images)
paper_target = [0 for i in range(0, paper_amount)]
# paper_target = np.array(paper_target)

# sicer : target=1
sicer_images = []
for x in range(sicer_amount):
    img = cv2.imread('./sicer/sicer_%03d.jpg' % (x))
    # img = cv2.imread('./sicer_000/sicer_%03d.jpg' % (x))
    img = cv2.resize(img, (300, 300))
    sicer_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# sicer_images = np.array(sicer_images)
sicer_target = [1 for i in range(0, sicer_amount)]
# sicer_target = np.array(sicer_target)

# stone : target=2
stone_images = []
for x in range(stone_amount):
    img = cv2.imread('./stone/stone_%03d.jpg' % (x))
    # img = cv2.imread('./stone_000/stone_%03d.jpg' % (x))
    img = cv2.resize(img, (300, 300))
    sicer_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# stone_images = np.array(stone_images)
stone_target = [2 for i in range(0, stone_amount)]
# stone_target = np.array(stone_target)

# merge sets
images = list(paper_images) + list(sicer_images) + list(stone_images)
target = list(paper_target) + list(sicer_target) + list(stone_target)
images = np.array(images)
target = np.array(target)

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

### 30 K-means
k = 30
voc, variance = kmeans(descriptors, k, 1)

### generate features
# 對於訓練集中的每一張圖片，統計vocabulary中K個word的“詞頻”，得到相應的直方圖
im_features = np.zeros((data_amount, k), 'float32')
for i in range(data_amount):
    words, distance = vq(des_list[i], voc)
    for w in words:
        im_features[i][w] += 1

# Standardization 平均&變異數標準化
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

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
# 20 pictures for prediction

prediction_images = []
# paper
for x in range(10):
    # img = cv2.imread('./prediction/paper_%03d.jpg' % (x))
    img = cv2.imread('./paper_000/paper_%03d.jpg' % (x))

    img = cv2.resize(img, (300, 300))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# sicer
for x in range(10):
    # img = cv2.imread('./prediction/sicer_%03d.jpg' % (x))
    img = cv2.imread('./sicer_000/sicer_%03d.jpg' % (x))
    img = cv2.resize(img, (300, 300))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# stone
for x in range(10):
    # img = cv2.imread('./prediction/stone_%03d.jpg' % (x))
    img = cv2.imread('./stone_000/stone_%03d.jpg' % (x))
    img = cv2.resize(img, (300, 300))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

prediction_images = np.array(prediction_images)
prediction_target = [0]*10 + [1]*10 + [2]*10
prediction_target = np.array(prediction_target)

prediction_des_list = []
for img in prediction_images:
    kpts = sift.detect(img)
    kpts, des = sift.compute(img, kpts)
    prediction_des_list.append(des)

prediction_descriptors = prediction_des_list[0]
for descriptor in prediction_des_list[1:]:
    descriptors = np.vstack((prediction_descriptors, descriptor))

### 20 K-means
k = 30
voc, variance = kmeans(descriptors, k, 1)

### generate features
prediction_im_features = np.zeros((30, k), 'float32')
for i in range(30):
    words, distance = vq(prediction_des_list[i], voc)
    for w in words:
        prediction_im_features[i][w] += 1

stdSlr = StandardScaler().fit(prediction_im_features)
prediction_im_features = stdSlr.transform(prediction_im_features)

print("Predictin Accuracy")
print(clf.score(prediction_im_features, prediction_target))


# print(clf.predict(x_train))
# print(clf.predict(x_test))
print(clf.predict(prediction_im_features))
