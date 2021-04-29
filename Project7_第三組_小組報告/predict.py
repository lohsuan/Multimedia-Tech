import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np
import cv2
import joblib
import function

# prediction
# paper_amount = 10
# sicer_amount = 10
# stone_amount = 10
# all_data_amount = 30
paper_amount = 150
sicer_amount = 150
stone_amount = 150
all_data_amount = 450
paper_images = function.img_read(paper_amount, './paper/paper_%03d.jpg')
sicer_images = function.img_read(sicer_amount, './sicer/sicer_%03d.jpg')
stone_images = function.img_read(stone_amount, './stone/stone_%03d.jpg')

# paper_images = function.img_read(paper_amount, './prediction/paper_%03d.jpg')
# sicer_images = function.img_read(sicer_amount, './prediction/sicer_%03d.jpg')
# stone_images = function.img_read(stone_amount, './prediction/stone_%03d.jpg')


paper_target = function.create_target_list(paper_amount, 0)
sicer_target = function.create_target_list(sicer_amount, 1)
stone_target = function.create_target_list(stone_amount, 2)

images = paper_images + sicer_images + stone_images
targets = paper_target + sicer_target + stone_target
images = np.array(images)
targets = np.array(targets)

k=30
im_features = function.create_feature(images, k, all_data_amount)

clf = LinearSVC()
clf = joblib.load("clf1.pkl")

print("Prediction Accuracy: ", clf.score(im_features, targets))
print(clf.predict(im_features))
