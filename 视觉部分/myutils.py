import cv2
import numpy as np

# 显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#图像缩放
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 初始化缩放比例，并获取图像尺寸
    dim = None
    (h, w) = image.shape[:2]

    # 如果宽度和高度均为0，则返回原图
    if width is None and height is None:
        return image

    # 宽度是0
    if width is None:
        # 根据高度计算缩放比例
        r = height / float(h)
        dim = (int(w * r), height)

    # 宽度不为0
    else:
        # 根据宽度计算缩放比例
        r = width / float(w)
        dim = (width, int(h * r))

    # 缩放图像
    resized = cv2.resize(image, dim, interpolation=inter)

    # 返回缩放后的图像
    return resized

#图像变换
def four_point_transform(image, pts):
    #pts为四个点坐标，按顺序为左上角，右上角，右下角，左下角

    # 获取输入坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标
    # 左上角点坐标和右下角点坐标
    s = pts.sum(axis=1)
    # rect[0] = pts[np.argmin(s)]
    # rect[2] = pts[np.argmax(s)]
    #
    # # 右上角点坐标和左下角点坐标
    # diff = np.diff(pts, axis=1)
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]

    center = pts.sum(axis=0) / 4
    cx, cy = center

    for position in pts:
        x, y = position
        if x < cx and y < cy:
            rect[0] = position
        elif x > cx and y < cy:
            rect[1] = position
        elif x > cx and y > cy:
            rect[2] = position
        elif x < cx and y > cy:
            rect[3] = position



    # 得到正确顺序的坐标
    (tl, tr, br, bl) = rect
    print('pts:')
    print(pts)
    print(rect)

    # 计算原图的宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算原图的高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 用原图的w和h，得到变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)

    # 进行变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped

#方形角点排序
def order_points(pts):
    """将四个点按 左上、右上、右下、左下 顺序排列"""
    rect = np.zeros((4, 2), dtype=int)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上（x+y最小）
    rect[2] = pts[np.argmax(s)]  # 右下（x+y最大）
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上（y-x最小）
    rect[3] = pts[np.argmax(diff)]  # 左下（y-x最大）
    return rect

