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

paper_amount = 50
sicer_amount = 50
stone_amount = 50
all_data_amount = 150

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

    print(descriptors)
    ### K-means
    voc, variance = kmeans(descriptors, k, 1)
    print(voc)

    ### generate features
    # 對於訓練集中的每一張圖片，統計vocabulary中K個word的“詞頻”，得到相應的直方圖
    im_features = np.zeros((all_data_amount, k), 'float32')
    for i in range(all_data_amount):
        words, distance = vq(des_list[i], voc)
        for w in words:
            im_features[i][w] += 1
    # print(im_features[0])

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

# data prepare
paper_images = img_read(paper_amount, './paper/paper_%03d.jpg')
sicer_images = img_read(sicer_amount, './sicer/sicer_%03d.jpg')
stone_images = img_read(stone_amount, './stone/stone_%03d.jpg')

paper_target = create_target_list(paper_amount, 0)
sicer_target = create_target_list(sicer_amount, 1)
stone_target = create_target_list(stone_amount, 2)

images = paper_images + sicer_images + stone_images
targets = paper_target + sicer_target + stone_target
images = np.array(images)
targets = np.array(targets)

k=30
im_features = create_feature(images, k, all_data_amount)

x_train, x_test, y_train, y_test = train_test_split(
                        im_features , targets,
                        test_size=0.25, random_state=0)

clf = fit_module(x_train, y_train)
save_module(clf, "clf1.pkl")
# clf = LinearSVC()
# clf = joblib.load("clf1.pkl")

print("Train Data Accuracy: ", clf.score(x_train, y_train))
print("Test Data Accuracy: ", clf.score(x_test, y_test))

# prediction
paper_amount = 50
sicer_amount = 50
stone_amount = 50
all_data_amount = 150
paper_images = img_read(paper_amount, './paper/paper_%03d.jpg')
sicer_images = img_read(sicer_amount, './sicer/sicer_%03d.jpg')
stone_images = img_read(stone_amount, './stone/stone_%03d.jpg')

# paper_images = img_read(paper_amount, './prediction/paper_%03d.jpg')
# sicer_images = img_read(sicer_amount, './prediction/sicer_%03d.jpg')
# stone_images = img_read(stone_amount, './prediction/stone_%03d.jpg')


paper_target = create_target_list(paper_amount, 0)
sicer_target = create_target_list(sicer_amount, 1)
stone_target = create_target_list(stone_amount, 2)

images = paper_images + sicer_images + stone_images
targets = paper_target + sicer_target + stone_target
images = np.array(images)
targets = np.array(targets)

k=30
im_features = create_feature(images, k, all_data_amount)

# clf = LinearSVC()
# clf = joblib.load("clf1.pkl")

print("Prediction Accuracy: ", clf.score(im_features, targets))
# print(clf.predict(im_features))

