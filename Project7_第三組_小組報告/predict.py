import numpy as np
import cv2
import joblib
import function
from skimage.feature import hog

win = cv2.imread('./win.jpg')
lose = cv2.imread('./lose.jpg')
tie = cv2.imread('./tie.jpg')

# human
# my_path = input('Please input your img path: ')
my_path = './paper.jpg'
my_img = cv2.imread(my_path)
my_img = cv2.resize(my_img, (300,300))
my_img_gray = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY) 
fd, hog_img = hog(my_img_gray,
                orientations=8, pixels_per_cell=(8,8),
                cells_per_block=(3,3), visualize=True,
                multichannel=False)
# im_features = function.to_hog(np.array(my_img_gray))
clf = joblib.load("hog.pkl")
my_predict_result = clf.predict(fd)

# Robot
image = function.random_image('./prediction_000/')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd, hog_img = hog(image_gray,
                orientations=8, pixels_per_cell=(8,8),
                cells_per_block=(3,3), visualize=True,
                multichannel=False)
# im_features = function.to_hog(np.array(image_gray))
predict_result = clf.predict(fd)

outcome = ''
if(predict_result[0] ==0 & my_predict_result[0] ==1):
    result = np.hstack((image,win,my_img))
elif(predict_result[0] ==0 & my_predict_result[0] ==2):
    result = np.hstack((image,lose,my_img))
elif(predict_result[0] ==1 & my_predict_result[0] ==2):
    result = np.hstack((image,win,my_img))
elif(predict_result[0] ==1 & my_predict_result[0] ==0):
    result = np.hstack((image,lose,my_img))
elif(predict_result[0] ==2 & my_predict_result[0] ==0):
    result = np.hstack((image,win,my_img))
elif(predict_result[0] ==2 & my_predict_result[0] ==1):
    result = np.hstack((image,lose,my_img))
else:
    result = np.hstack((image,tie,my_img))


cv2.imshow('robot', image)
cv2.imshow('me', my_img)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()