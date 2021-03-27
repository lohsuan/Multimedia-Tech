import numpy as np
import cv2
# coin2
img_3 = cv2.imread("./images/coin2.jpg")
img = cv2.imread("./images/coin2.jpg", 0)
img_gray = cv2.imread("./images/coin2.jpg", 0)

height = img.shape[0]
width = img.shape[1]
price = 0

img = cv2.GaussianBlur(img, (55, 55), 5)
ret, img  = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

for i in stats[:, ]:
    if(i[4] > 1000000 and i[4] <2000000):
        x1 = int(i[0])
        y1 = int(i[1])
        x2 = int(i[0] + i[2])
        y2 = int(i[1] + i[3])
        if(i[2] > 1000):
            price += 500
            cv2.rectangle(img_3, (x1, y1), (x2, y2), (255, 0, 255), 10)
        elif(i[4] > 1400000):
            price += 1000
            cv2.rectangle(img_3, (x1, y1), (x2, y2), (255, 255, 255), 10)
        else:
            price += 100
            cv2.rectangle(img_3, (x1, y1), (x2, y2), (255, 0, 0), 10)

circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 250, param1=700, param2=50,minRadius=100, maxRadius=160)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]

        # draw rectangle
        # 1
        if(i[2] < 120):
            price += 1
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]),
                          (i[0]+i[2], i[1]-i[2]), (0, 0, 255), 10)
        # 5
        elif(i[2] < 130):
            price += 5
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]),
                          (i[0]+i[2], i[1]-i[2]), (0, 125, 255), 10)
        # 50
        elif(i[2] > 150):
            price += 50
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]),
                          (i[0]+i[2], i[1]-i[2]), (0, 255, 0), 10)
        # 10
        else:
            price += 10
            cv2.rectangle(img_3, (i[0]-i[2], i[1]+i[2]),
                          (i[0]+i[2], i[1]-i[2]), (0, 255, 255), 10)

print("Total money is: " + str(price))

img_3 = cv2.resize(img_3, (int(width*0.25), int(height*0.25)), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("./outcome/coin2_out.jpg", img_3)
