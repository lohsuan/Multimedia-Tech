import numpy as np
import cv2

img_3 = cv2.imread("./images/coin.jpg")
img = cv2.imread("./images/coin.jpg", 0)
height = img.shape[0]
width = img.shape[1]
# img = cv2.resize(img, (int(width*0.25), int(height*0.25)), interpolation=cv2.INTER_CUBIC)
# edged = cv2.Canny(img, 30, 150)
# img = cv2.dilate(img, np.ones((3, 3)), iterations=1)
# img = cv2.erode(img, np.ones((3, 3)), iterations=1)
# img = cv2.erode(img, np.ones((3, 3)), iterations=3)
# img = cv2.dilate(img, np.ones((3, 3)), iterations=4)
# img = cv2.erode(img, np.ones((3, 3)), iterations=2)

# ret, img  = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 360, param1=100, param2=30,minRadius=180, maxRadius=300)
# if circles is not None:
#     print("Total circle detect: " + str(len(circles[0])))

#  draw circles
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         center = (i[0], i[1])
#         radius = i[2]
#         cv2.circle(img, center, radius, (255, 255, 255), -3)

# cv2.imwrite("./coin_out1.jpg", img)

price = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # center = (i[0], i[1])
        # radius = i[2]
        # cv2.circle(img_3, center, radius, (255, 255, 255), -3)
        # print(i[2])

        # draw rectangle
        # 1
        if(i[2]<190):
            price += 1
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]), (i[0]+i[2], i[1]-i[2]), (0,0,255), 10)
        # 5
        elif(i[2] < 210):
            price += 5
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]), (i[0]+i[2], i[1]-i[2]), (0, 125, 255), 10)
        # 50
        elif(i[2] > 248):
            price += 50
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]), (i[0]+i[2], i[1]-i[2]), (0, 255, 0), 10)
        # 10
        else:
            price += 10
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]),
            (i[0]+i[2], i[1]-i[2]), (0, 255, 255), 10)

print("Total money is: " + str(price))
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
# print(num_labels)
# blurred = cv2.GaussianBlur(img, (11, 11), 10)
# ret, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)

img_3 = cv2.resize(img_3, (int(width*0.25), int(height*0.25)), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("./outcome/coin_out.jpg", img_3)
