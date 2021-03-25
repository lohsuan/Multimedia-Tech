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

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 45, param1=100,
param2=30, minRadius=200, maxRadius=500)


# let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8U);
# let circles = new cv.Mat();
# let color = new cv.Scalar(255, 0, 0);
# cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
# // You can try more different parameters
# cv.HoughCircles(src, circles, cv.HOUGH_GRADIENT,
#                 1, 45, 75, 40, 0, 0);
#draw circles
for i in circles:
    x = circles.data32F[i * 3];
    y = circles.data32F[i * 3 + 1];
    radius = circles.data32F[i * 3 + 2];
    center = cv2.Point(x, y);
    cv2.circle(dst, center, radius, color);
# for (let i = 0; i < circles.cols; ++i) {
#     let x = circles.data32F[i * 3];
#     let y = circles.data32F[i * 3 + 1];
#     let radius = circles.data32F[i * 3 + 2];
#     let center = new cv.Point(x, y);
#     cv.circle(dst, center, radius, color);
# }


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
print(num_labels)
print(circles)
# print((circles))
# blurred = cv2.GaussianBlur(img, (11, 11), 10)

cv2.imwrite("./coin_out3.jpg", img)
