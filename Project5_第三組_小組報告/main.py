# from skimage import data, io, filters
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

# data preparing

# people
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)

people_x_images = lfw_people.images[0:1000]
people_y_target = [0 for i in range(0, 1000)]
people_y_target = np.array(people_y_target)

print(people_x_images.shape)
print(people_y_target.shape)

# not people

cats_train_amount = 1000
cats_test_amount = 489
prediction_amount = 10

cats_x_train = []
for x in range(cats_train_amount):
    img = cv2.imread('./cats_train_0_999/cat.%d.jpg' % (x))
    img = cv2.resize(img, (37, 50))
    cats_x_train.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cats_x_train = np.array(cats_x_train)
# print(cats_x_train.shape)

cv2.imshow("test",cats_x_train[990])
cv2.waitKey(0)
cats_y_train = [1 for i in range(0, 1000)]


# hog

# people_hogged_fd = []
# people_hogged_img = []

# for image in people_images:
#   fd, hog_img = hog(image,
#             orientations=8, pixels_per_cell=(8,8),
#             cells_per_block=(3,3), visualize=True,
#             transform_sqrt, multichannel=True
#             )
#   people_hogged_fd.append(fd)         # data we want
#   people_hogged_img.append(hog_img)

# x_train, x_test, y_train, y_test = train_test_split(
#                         people_hogged_fd, people_target,
#                         test_size=0.3, random_state=0)

# # svm

# clf = svm.SVC(C=1, kernel='linear', gamma='scale')
# clf.fit(x_train, y_train)

# print("Accuracy")
# print(clf.score(X_train, y_train))
# print(clf.score(X_test, y_test))