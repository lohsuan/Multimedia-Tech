import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.processing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from imutils import paths
# cats: https://drive.google.com/drive/folders/1lDoaCT78dJYHxFhlBZVvNSiktN6HIN_J?usp=sharing
# prediction_1990_1999: https://drive.google.com/drive/folders/13gwUP4-we6yhHGDE3j0fnLkhz10cBYu0?usp=sharing

### data preparing

# train_path_cars = "./cars/"
# train_path_cats = "./cats/"
# training_names_cars = os.listdir(train_path_cars)
# training_names_cats = os.listdir(train_path_cats)

# cars_img_paths = []
# cars_img_classes = []
# cars_class_id = 0
# for training_name in training_names_cars:
#   dir = os.path.join(train_path_cars, training_name)
#   class_path = list(path.list_images(dir))
#   image_paths+=class_path
#   image_classes +=

# cars : target=0
cars_amount = 1000
cars_images = []
for x in range(cars_amount):
    img = cv2.imread('./cars/%05d.jpg' % (x+1))
    img = cv2.resize(img, (60, 40))
    cars_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cars_images = np.array(cars_images)
cars_target = [0 for i in range(0, 1000)]
cars_target = np.array(cars_target)

# cats : target=1
cats_amount = 1000
cats_images = []
for x in range(cats_amount):
    img = cv2.imread('./cats/cat.%d.jpg' % (x))
    img = cv2.resize(img, (60, 40))
    cats_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cats_images = np.array(cats_images)
cats_target = [1 for i in range(0, 1000)]
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
    img = cv2.resize(img, (60, 40))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# cats
for x in range(10):
    img = cv2.imread('./prediction/cat.%d.jpg' % (x+1990))
    img = cv2.resize(img, (60, 40))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

prediction_images = np.array(prediction_images)
prediction_target = [0]*10 + [1]*10
prediction_target = np.array(prediction_target)

### SIFT

images_kp = []
sift = cv2.SIFT_create()

for img in images:
  kp, des = sift.detectAndCompute(img, None)
  images_kp.append(kp)

x_train, x_test, y_train, y_test = train_test_split(
                        images_kp, target,
                        test_size=0.25, random_state=0)

### svm

clf = LinearSVC()
clf.fit(x_train, y_train)

### accuracy
print("Train Data Accuracy")
print(clf.score(x_train, y_train))
print("Test Data Accuracy")
print(clf.score(x_test, y_test))

### 20 Prediction test
prediction_images_hogged_kp = []
for img in prediction_images:
  kp, des = sift.detectAndCompute(img, None)
  prediction_images_hogged_kp.append(kp)

print("Predictin Accuracy")
print(clf.score(prediction_images_hogged_kp, prediction_target))

