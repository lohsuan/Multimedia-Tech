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


# 注意，这里我们对图像进行裁剪，不能使用图像缩放resize。
# 因为我们在采样的过程中，根据我们的算法，是在top和left两个方向增加了数值
# 所以整体图像是像右下方移动的
# 所以我们需要把我们重采样过程中top和left方向上多加的像素给减掉
# 如果只是单纯的resize，会发现图片整体和原图虽然大小一致，
# 但是图片中物体的位置是移动了的
# 因此这个操作是图像裁剪，与我们重采样时扩边操作对应
# 而不是图像的缩放
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


apple = cv2.imread("./apple.jpg")
orange = cv2.imread("./orange.jpg")

gpL = 6

# 为苹果建立高斯金字塔，包括原图共7层
gpA = [apple]
for i in range(gpL):
    g = cv2.pyrDown(gpA[i])
    gpA.append(g)

# 为桔子建立高斯金字塔，包括原图共7层
gpO = [orange]
for i in range(gpL):
    g = cv2.pyrDown(gpO[i])
    gpO.append(g)

# 基于高斯金字塔建立苹果的拉普拉斯金字塔
# 注意这里在建完金字塔后，将金字塔的第一层
# 用高斯金字塔的倒数第二层(第五层)替代了
# 这样才能再后来重新上采样的时候还原原图
# 否则只靠拉普拉斯金字塔的信息上采样是无法
# 还原原图的，只会得到黑乎乎的图像
# 而之所以用第5层，是因为建拉普拉斯金字塔
# 只能从高斯金字塔的导数第二层建起，而
# 倒数第二层对应索引是5
# 当然其实还有一种方案，那就是不替代拉普拉斯金字塔
# 的第一层，而是在第一层的前面再加一层高斯
# 金字塔。这样得到的拉普拉斯金字塔一共是7层
# 包括一层高斯金字塔的最后一层和6层正常的拉普拉斯
# 金字塔。但这样可能会给后面的范围带来些麻烦，
# 而且多一层对于融合效果提升不大，所以这里不采用这个方法。
# 当然可能会好奇，为什么一定要用
# 高斯金字塔的最后一层。前面也说了，正是因为
# 需要恢复原图信息，必须要有原图，否则是无法还原
# 原图信息的。高斯金字塔包含原图信息。
lpA = []
for i in range(gpL, 0, -1):
    l = getLplacianSub(gpA[i - 1], cv2.pyrUp(gpA[i]))
    lpA.append(l)
lpA[0] = gpA[-2]

# 基于高斯金字塔建立桔子的拉普拉斯金字塔
# 同时注意拉普拉斯金字塔的第一层的替换
lpO = []
for i in range(gpL, 0, -1):
    l = getLplacianSub(gpO[i - 1], cv2.pyrUp(gpO[i]))
    lpO.append(l)
lpO[0] = gpO[-2]

# 对于每一层金字塔将两个图像各取一半合并
# 这里用到了Numpy的hstack函数，表示将两个矩阵合并到一起，注意不是加
# 其中h表示水平方向，v表示竖直方向
# 例如A为3*3，B为3*3，hstack合并后矩阵大小是3*6
lpAO = []
for i in range(gpL):
    rows, cols, bpt = lpA[i].shape
    lp = np.hstack((lpA[i][:, :cols // 2, :], lpO[i][:, cols // 2:, :]))
    lpAO.append(lp)

# 逐步上采样重建金字塔
# 注意这里的迭代变量关系
# 这里是拉普拉斯金字塔第一层上采样到与第二层大小一致，
# 然后与现有的第二层相加，得到新的第二层
# 再将这个新的第二层上采样，和第三层相加
# 所以需要注意并不是拿老的第二层与第三层相加
# 否则是得不到正确结果的
gao = lpAO[0]
for i in range(gpL - 1):
    gao = cv2.pyrUp(gao)
    gao = getLplacianAdd(lpAO[i + 1], gao)

# 由于重采样回来后图像变大了，所以可以重新裁剪到原图大小
gao = restoreXY(apple, gao)

# 直接混合对比
# direct = np.hstack((apple[:, :apple.shape[1] // 2], orange[:, orange.shape[1] // 2:]))
# cv2.imwrite("./fruit_out_direct.jpg", direct)

cv2.imwrite("./fruit_out.jpg", gao)
