import numpy as np
import cv2
# circle
img = cv2.imread("./coin.jpg", 0)
# height = img.shape[0]
# width = img.shape[1]
# img = cv2.resize(img, (int(width*0.25), int(height*0.25)), interpolation=cv2.INTER_CUBIC)
edged = cv2.Canny(img, 30, 150)
img = cv2.dilate(img, np.ones((3, 3)), iterations=2)
img = cv2.erode(img, np.ones((3, 3)), iterations=2)
ret, img  = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)



# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
# print(num_labels)
# blurred = cv2.GaussianBlur(img, (11, 11), 10)


cv2.imwrite("./coin_out3.jpg", img)
