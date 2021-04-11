# from skimage import data, io, filters
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

# requirement
# pip install scikit-datasets

# data preparing
# people
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.4)

# print(lfw_people.keys())
# dict_keys(['data', 'images', 'target', 'target_names', 'DESCR'])

people_images = lfw_people.images
people_target = [0 for i in range(0, 1140)]
people_target = np.array(people_target)

print(people_images.shape)
print(people_target.shape)

# not people

train_folder = r'./train/buildings'
test_folder = r'./test/buildings'


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
