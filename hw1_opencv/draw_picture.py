import numpy as np
import cv2

img = cv2.imread("./cat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# img = np.zeros((400, 400, 3), np.uint8)
# img.fill(200)

cv2.circle(img, (300, 300), 30, (255, 0, 255), 3, 1)
cv2.line(img, (300, 70), (480, 220), (0, 255, 0), 5)
cv2.rectangle(img, (10, 0), (580, 320), (255, 200, 100), 5)
cv2.rectangle(img, (20, 20), (100, 100), (0, 255, 255), -1)
cv2.circle(img, (510, 130), 70, (255, 255, 0), 5)
cv2.putText(img, "I can fly !!!", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200,
200), 3)
cv2.imwrite( "./picture1.jpg", img)

# show pictures
# cv2.imshow( "show_picture", img)
# cv2. waitKey(1000)