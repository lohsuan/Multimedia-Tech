import numpy as np
import cv2

img_3 = cv2.imread("./images/coin.jpg")
img = cv2.imread("./images/coin.jpg", 0)
height = img.shape[0]
width = img.shape[1]

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 360, param1=100, param2=30,minRadius=180, maxRadius=300)

price = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:        
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

img_3 = cv2.resize(img_3, (int(width*0.25), int(height*0.25)), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("./outcome/coin_out.jpg", img_3)
