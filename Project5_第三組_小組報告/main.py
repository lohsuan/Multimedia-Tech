import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import cv2

# requirement
# pip install scikit-datasets
# images' dataset : https://www.kaggle.com/c/dogs-vs-cats/data?select=sampleSubmission.csv
# cats_0_999: https://drive.google.com/drive/folders/1lDoaCT78dJYHxFhlBZVvNSiktN6HIN_J?usp=sharing
# prediction_1990_1999: https://drive.google.com/drive/folders/13gwUP4-we6yhHGDE3j0fnLkhz10cBYu0?usp=sharing

### data preparing

# people
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)

people_images = lfw_people.images[:1000]
people_target = [0 for i in range(0, 1000)]
people_target = np.array(people_target)

# not people
cats_amount = 1000
cats_images = []
for x in range(cats_amount):
    img = cv2.imread('./cats_0_999/cat.%d.jpg' % (x))
    img = cv2.resize(img, (37, 50))
    cats_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cats_images = np.array(cats_images)
cats_target = [1 for i in range(0, 1000)]
cats_target = np.array(cats_target)

# merge two sets
images = list(people_images) + list(cats_images)
target = list(people_target) + list(cats_target)
images = np.array(images)
target = np.array(target)

# 20 pictures for prediction 
prediction_images = lfw_people.images[1000:1010]
prediction_images = list(prediction_images)

for x in range(10):
    img = cv2.imread('./prediction_1990_1999/cat.%d.jpg' % (x+1990))
    img = cv2.resize(img, (37, 50))
    prediction_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

prediction_images = np.array(prediction_images)
prediction_target = [0]*10 + [1]*10
prediction_target = np.array(prediction_target)

### hog

images_fd = []
for img in images:
  fd, hog_img = hog(img,
            orientations=8, pixels_per_cell=(8,8),
            cells_per_block=(3,3), visualize=True,
            multichannel=False)
  images_fd.append(fd)

x_train, x_test, y_train, y_test = train_test_split(
                        images_fd, target,
                        test_size=0.25, random_state=0)

### svm

clf = svm.SVC(C=1, kernel='linear', gamma='scale')
clf.fit(x_train, y_train)

### accuracy 
print("Train Data Accuracy")
print(clf.score(x_train, y_train))
print("Test Data Accuracy")
print(clf.score(x_test, y_test))

### 20 Prediction test
prediction_images_hogged_fd = []
for img in prediction_images:
  fd, hog_img = hog(img,
                    orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), visualize=True,
                    multichannel=False)
  prediction_images_hogged_fd.append(fd)

print("Predictin Accuracy")
print(clf.score(prediction_images_hogged_fd, prediction_target))
