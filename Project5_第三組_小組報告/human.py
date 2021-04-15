# from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn import svm

import numpy as np
from scipy import signal
import scipy.misc

def s_x(img):
    kernel = np.array([[-1, 0, 1]])
    imgx = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    return imgx
def s_y(img):
    kernel = np.array([[-1, 0, 1]]).T
    imgy = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    return imgy

def grad(img):
    imgx = s_x(img)
    imgy = s_y(img)
    s = np.sqrt(imgx**2 + imgy**2)
    theta = np.arctan2(imgx, imgy) #imgy, imgx)
    theta[theta<0] = np.pi + theta[theta<0]
    return (s, theta)

import time
from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=123)
timestamp1 = time.time()
clf = SVC(C=1, kernel='linear')
clf.fit(Xtrain, ytrain)
print("%d support vectors out of %d points" % (len(clf.support_vectors_), len(Xtrain)))
timestamp2 = time.time()
print("sklearn LinearSVC took %.2f seconds" % (timestamp2 - timestamp1))
ypred = clf.predict(Xtest)
print('accuracy', accuracy_score(ytest, ypred))

# sklearn.datasets.fetch_lfw_people(*, data_home=None, funneled=True, resize=0.5,
#                 min_faces_per_person=0, color=False, slice_=slice(70, 195, None),
#                 slice(78, 172, None), download_if_missing=True, return_X_y=False)

# Project4_第三組_小組報告
# wine = datasets.load_wine()
#
# X = wine.data
# y = wine.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# clf = svm.SVC(C=1, kernel='linear', gamma='auto')
# clf.fit(X_train, y_train) # 將訓練集送入訓練(fit)
#
# # print("predict")
# # print(clf.predict(X_train)) #target=y_train
# # print(clf.predict(X_test))  #target=y_test
#
# # clf.score(data, target)
# # 輸出"以data進行predict後的結果"與"target進行比對"計算準確率
# print("Accuracy")
# print(clf.score(X_train, y_train))
# print(clf.score(X_test, y_test))
