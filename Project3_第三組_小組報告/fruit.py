import numpy as np
import cv2

def getLplacianAdd(gauss, g):
    gauss_width = gauss.shape[0]
    gauss_height = gauss.shape[1]
    g_width = g.shape[0]
    g_height = g.shape[1]
    d_width = g_width - gauss_width
    d_height = g_height - gauss_height
    if d_width != 0 or d_height != 0:
        temp = cv2.copyMakeBorder(
            gauss, d_width, 0, d_height, 0, cv2.BORDER_REPLICATE)
    else:
        temp = gauss
    layer = cv2.add(temp, g)
    return layer


def getLplacianSub(gauss, g):
    gauss_width = gauss.shape[0]
    gauss_height = gauss.shape[1]
    g_width = g.shape[0]
    g_height = g.shape[1]
    d_width = g_width - gauss_width
    d_height = g_height - gauss_height
    if d_width != 0 or d_height != 0:
        temp = cv2.copyMakeBorder(
            gauss, d_width, 0, d_height, 0, cv2.BORDER_REPLICATE)
    else:
        temp = gauss
    layer = cv2.subtract(temp, g)
    return layer

def restoreXY(origin, present):
    ori_row = origin.shape[0]
    ori_col = origin.shape[1]
    pre_row = present.shape[0]
    pre_col = present.shape[1]
    if ori_row != pre_row or ori_col != pre_col:
        result = present[pre_row - ori_row:, pre_col - ori_col:]
    else:
        result = present
    return result

apple = cv2.imread("./images/apple.jpg")
orange = cv2.imread("./images/orange.jpg")

gpL = 6

# apple
gpA = [apple]
for i in range(gpL):
    g = cv2.pyrDown(gpA[i])
    gpA.append(g)

# orange
gpO = [orange]
for i in range(gpL):
    g = cv2.pyrDown(gpO[i])
    gpO.append(g)

lpA = []
for i in range(gpL, 0, -1):
    l = getLplacianSub(gpA[i - 1], cv2.pyrUp(gpA[i]))
    lpA.append(l)
lpA[0] = gpA[-2]

lpO = []
for i in range(gpL, 0, -1):
    l = getLplacianSub(gpO[i - 1], cv2.pyrUp(gpO[i]))
    lpO.append(l)
lpO[0] = gpO[-2]

lpAO = []
for i in range(gpL):
    rows, cols, bpt = lpA[i].shape
    lp = np.hstack((lpA[i][:, :cols // 2, :], lpO[i][:, cols // 2:, :]))
    lpAO.append(lp)

gao = lpAO[0]
for i in range(gpL - 1):
    gao = cv2.pyrUp(gao)
    gao = getLplacianAdd(lpAO[i + 1], gao)

gao = restoreXY(apple, gao)

cv2.imwrite("./outcome/fruit_out.jpg", gao)
