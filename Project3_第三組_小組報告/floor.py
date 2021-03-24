import numpy as np
import cv2
# circle
img_3 = cv2.imread("./images/floor.jpg")   # 3 channels
img = cv2.imread("./images/floor.jpg", 0)
height = img.shape[0]
width = img.shape[1]

img_blurred = cv2.GaussianBlur(img_3, (99, 99), 5)
ret, img  = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
img = cv2.GaussianBlur(img, (99, 99), 5)
ret, img  = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
# img = cv2.dilate(img, np.ones((3, 3)), iterations=1)
# img = cv2.erode(img, np.ones((3, 3)), iterations=1)

# img = cv2.Canny(img, 50, 200, None, 3)


img = 255 - img
# cv2.imwrite("./floor_out.jpg", img)

# lines = cv2.HoughLinesP(img, 1, np.pi/180, 200, 100, 30)
lines = cv2.HoughLinesP(img, 1, np.pi/180, 800, 50, 100)

# print(lines.shape)

for i in lines [:, 0, :]:
    img_3 = cv2.line(img_blurred, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 3)

img_3 = cv2.resize(img_3, (int(width*0.25), int(height*0.25)),interpolation=cv2.INTER_CUBIC)

cv2.imwrite("./outcome/floor_out.jpg", img_3)
