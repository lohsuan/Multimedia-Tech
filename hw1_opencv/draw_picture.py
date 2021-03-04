import numpy as np
import cv2

img = np.zeros((400, 400, 3), np.uint8)
img.fill(200)

cv2.line(img, (0, 0), (255, 255), (0, 0, 255), 5)
cv2.rectangle(img, (0, 0), (255, 255), (0, 255, 0), 5)
cv2.rectangle(img, (0, 0), (100, 100), (0, 255, 255), -5)
cv2.circle(img, (100, 100), 100, (255, 255, 0), 5)
cv2.circle(img, (100, 100), 50, (255, 0, 255), -1)

cv2.imwrite( "./picture.jpg", img)

# show pictures
cv2.imshow( "show_picture", img)
cv2. waitKey(1000)