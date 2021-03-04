import numpy as np
import cv2


img = cv2.imread("./cat.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# cv2.imshow( "show_img_gray", img_gray)
#　cv2.imshow( "show_img_hsv", img_hsv)
# cv2.imshow( "show_img_ycrcb", img_ycrcb)

cv2.imwrite( "./img_gray.jpg", img_gray)
cv2.imwrite( "./img_hsv.jpg", img_hsv)
cv2.imwrite( "./img_ycrcb.jpg", img_ycrcb)

height = img.shape[0]
width  = img.shape[1]

img_R = np.zeros((height, width, 3), np.uint8)
img_G = np.zeros((height, width, 3), np.uint8)
img_B = np.zeros((height, width, 3), np.uint8)

img_R[:, :, 2] = img[:, :, 2]
img_G[:, :, 1] = img[:, :, 1]
img_B[:, :, 0] = img[:, :, 0]

cv2.imwrite( "./img_R.jpg", img_R)
cv2.imwrite( "./img_G.jpg", img_G)
cv2.imwrite( "./img_B.jpg", img_B)

img = np.zeros((400, 400, 3), np.uint8)
img.fill(200)

cv2.line(img, (0, 0), (255, 255), (0, 0, 255), 5)
cv2.rectangle(img, (0, 0), (255, 255), (0, 255, 0), 5)
cv2.rectangle(img, (0, 0), (100, 100), (0, 255, 255), -5)
cv2.circle(img, (100, 100), 100, (255, 255, 0), 5)
cv2.circle(img, (100, 100), 50, (255, 0, 255), -1)

cv2.imwrite( "./img.jpg", img)