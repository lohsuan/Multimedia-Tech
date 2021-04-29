import matplotlib.pyplot as plt
from skimage import data, exposure
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
import numpy as np
import cv2
import joblib

def img_read(data_amount, img_path) -> list:
    images = []
    for x in range(data_amount):
        img = cv2.imread(img_path % (x))
        img = cv2.resize(img, (300, 300))
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return images

def create_target_list(data_amount, target) -> list:
    targets = [target] *data_amount
    return targets

def create_feature(images, k, all_data_amount) ->list:
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
    voc, variance = kmeans(descriptors, k, 1)

    ### generate features
    # 對於訓練集中的每一張圖片，統計vocabulary中K個word的“詞頻”，得到相應的直方圖
    im_features = np.zeros((all_data_amount, k), 'float32')
    for i in range(all_data_amount):
        words, distance = vq(des_list[i], voc)
        for w in words:
            im_features[i][w] += 1

    # Standardization 平均&變異數標準化
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features) 
    return im_features

def fit_module(x_train, y_train):
    clf = LinearSVC(dual=True, max_iter = 3000)
    clf.fit(x_train, y_train)
    return clf

def save_module(clf, module_name):
    joblib.dump(clf, module_name) # 儲存模型檔案(svc為訓練的模型)
