from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage.feature import hog
import numpy as np
import cv2
import joblib
import os

def img_read(dir_path) -> list:
    data = []
    images = os.listdir(dir_path)   # open the dir
    for img in images:
        img = cv2.imread(dir_path + img)
        img = cv2.resize(img, (300,300))
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return data

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
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(descriptors)
    #voc, variance = kmeans(descriptors, k, 1, k_or_guess=0)
    #print(voc)

    ### generate features
    im_features = np.zeros((all_data_amount, k), 'float32')
    for i in range(all_data_amount):
        words, distance = vq(des_list[i], kmeans.cluster_centers_)
        for w in words:
            im_features[i][w] += 1

    # Standardization
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
    clf = svm.SVC(C=1, kernel='linear', gamma='scale') # for hog
    # clf = LinearSVC(dual=True, max_iter = 3000)   # for sift 
    clf.fit(x_train, y_train)
    return clf

def save_module(clf, module_name):
    joblib.dump(clf, module_name)

# data prepare
# paper_amount = 219
# sicer_amount = 195
# stone_amount = 173
# all_data_amount = 587
# paper_images = img_read('./paper_000/')
# sicer_images = img_read('./sicer_000/')
# stone_images = img_read('./stone_000/')
# paper_target = create_target_list(paper_amount, 0)
# sicer_target = create_target_list(sicer_amount, 1)
# stone_target = create_target_list(stone_amount, 2)
# # merge data
# images = paper_images + sicer_images + stone_images
# targets = paper_target + sicer_target + stone_target
# images = np.array(images)
# targets = np.array(targets)

# # im_features = create_feature(images, k=30, all_data_amount)
# im_features = to_hog(images)

# x_train, x_test, y_train, y_test = train_test_split(
#                         im_features , targets,
#                         test_size=0.25, random_state=0)

# clf = fit_module(x_train, y_train)
# save_module(clf, "hog.pkl")

# print("Train Data Accuracy: ", clf.score(x_train, y_train))
# print("Test Data Accuracy: ", clf.score(x_test, y_test))

# # prediction
# paper_amount = 10
# sicer_amount = 10
# stone_amount = 10
# all_data_amount = 30

# images = img_read('./prediction_000/')
# paper_target = create_target_list(paper_amount, 0)
# sicer_target = create_target_list(sicer_amount, 1)
# stone_target = create_target_list(stone_amount, 2)

# targets = paper_target + sicer_target + stone_target
# images = np.array(images)
# targets = np.array(targets)

# # im_features = create_feature(images, k=30, all_data_amount)
# im_features = to_hog(images)
# clf = joblib.load("hog.pkl")
# print(clf.predict(im_features))
# print("Prediction Accuracy: ", clf.score(im_features, targets))

my_path = './paper.jpg'
my_img = cv2.imread(my_path)
# cv2.imshow('1',my_img)
my_img = cv2.resize(my_img, (300,300))
# cv2.imshow('2',my_img)
my_img_gray = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY) 
# cv2.imshow('3',my_img_gray)

im_features = to_hog(my_img_gray)
clf = joblib.load("hog.pkl")
print(clf.predict(im_features))

# cv2.waitKey(0)
# cv2.destroyAllWindows()