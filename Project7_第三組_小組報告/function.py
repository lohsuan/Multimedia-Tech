# import matplotlib.pyplot as plt
# from skimage import data, exposure
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.vq import *
# from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage.feature import hog
import numpy as np
import cv2
import joblib
import os
import random

def img_read(dir_path) -> list:
    data = []
    images = os.listdir(dir_path)   # open the dir
    for img in images:
        print("reading ", dir_path + img)
        img = cv2.imread(dir_path + img)
        img = cv2.resize(img, (300,300))
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return data

def create_target_list(data_amount, target) -> list:
    targets = [target] *data_amount
    return targets

def create_feature(images, k, all_data_amount) -> list:
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

    ### K-means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(descriptors)

    ### generate features
    im_features = np.zeros((all_data_amount, k), 'float32')
    for i in range(all_data_amount):
        words, distance = vq(des_list[i], kmeans.cluster_centers_)
        for w in words:
            im_features[i][w] += 1

    # Standardization 平均&變異數標準化
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features) 
    return im_features

def to_hog(images) ->list: 
    images_fd = []
    for img in images:
        fd, hog_img = hog(img,
                    orientations=8, pixels_per_cell=(8,8),
                    cells_per_block=(3,3), visualize=True,
                    multichannel=False)
        images_fd.append(fd)
    return images_fd

def fit_module(x_train, y_train):
    clf = LinearSVC(dual=True, max_iter = 3000)
    clf.fit(x_train, y_train)
    return clf

def save_module(clf, module_name):
    joblib.dump(clf, module_name) # 儲存模型檔案(svc為訓練的模型)

def random_image(folder) -> list:
    # data = []
    images = os.listdir(folder)
    random_filename = random.choice(images)
    print(folder + random_filename)
    image = cv2.imread(folder + random_filename)
    image = cv2.resize(image, (300,300))
    # data.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return image