import numpy as np
import cv2


img = cv2.imread('cat.0.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

# images_fd.append(kp)
