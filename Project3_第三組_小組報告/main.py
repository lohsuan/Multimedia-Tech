import numpy as np
import cv2
# circle
img = cv2.imread("./circle.jpg", 0)

img = cv2.dilate(img, np.ones((7, 7)), iterations=3)
img = cv2.erode(img, np.ones((7, 7)), iterations=3)

ret, out = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

cv2.imwrite("./circle_out.jpg", out)

# man
img0 = cv2.imread("./man.jpg", 0)
img = cv2.imread("./man.jpg", 0)

img = cv2.dilate(img, np.ones((3, 3)), iterations=2)

ret, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
ret, img0 = cv2.threshold(img0, 170, 255, cv2.THRESH_BINARY)

out = img - img0

cv2.imwrite("./man_out.jpg", out)
