import numpy as np
import cv2
import joblib
import function
from skimage.feature import hog

win = cv2.imread('./win.jpg')
win = cv2.resize(win, (300, 300))
lose = cv2.imread('./lose.jpg')
lose = cv2.resize(lose, (300, 300))
tie = cv2.imread('./tie.jpg')
tie = cv2.resize(tie, (300, 300))

# human
my_path = input('Please input your img path: ')
my_img = function.one_img_read(my_path)
my_img = np.array(my_img)
im_features = function.to_hog(my_img)
clf = joblib.load("hog.pkl")
my_predict_result = clf.predict(im_features)
i1 = cv2.imread(my_path)
i1 = cv2.resize(i1, (300, 300))

# Robot
rawimage, imagepath = function.random_image('./prediction_000/')
gray = []
gray.append(cv2.cvtColor(rawimage, cv2.COLOR_BGR2GRAY))
image = np.array(gray)
im_features = function.to_hog(image)
predict_result = clf.predict(im_features)
i2 = cv2.imread(imagepath)
i2 = cv2.resize(i2, (300, 300))

if predict_result[0] == 0 and my_predict_result[0] == 1 :
    result = np.hstack((i2, win, i1))
elif predict_result[0] == 0 and my_predict_result[0] == 2 :
    result = np.hstack(i2, lose, i1)
elif predict_result[0] == 1 and my_predict_result[0] == 2 :
    result = np.hstack((i2, win, i1))
elif predict_result[0] == 1 and my_predict_result[0] == 0 :
    result = np.hstack((i2, lose, i1))
elif predict_result[0] == 2 and my_predict_result[0] == 0 :
    result = np.hstack((i2, win, i1))
elif predict_result[0] == 2 and my_predict_result[0] == 1 :
    result = np.hstack((i2, lose, i1))
else:
    result = np.hstack((i2, tie, i1))

cv2.imwrite("./result.jpg", result)
cv2.imshow('robot', rawimage)
cv2.imshow('me', my_img)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()