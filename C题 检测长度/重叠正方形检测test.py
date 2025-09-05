import cv2
import functions
import numpy as np

frame = cv2.imread(r"project\2025\imgs\question5.png")

target = functions.detect_rectangle(frame)

# 计算正外接矩形
x, y, w, h = cv2.boundingRect(target)

# 截取矩形区域
frame_cut = frame[y:y+h, x:x+w]
result = frame_cut.copy()


gray = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2GRAY)

otsu_thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("thresh", binary_image)

contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_area = 5000
approxs = []

for cnt in contours:
    # 6.1 面积筛选
    area = cv2.contourArea(cnt)
    print(f"轮廓面积: {area}")
    if area < min_area:
        continue

    # 近似轮廓 0.005 - 0.015
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approxs.append(approx)
    print(f"点数: {len(approx)}")
    
    # 绘制近似轮廓（红色）
    cv2.polylines(frame_cut, [approx.astype(np.int32)], True, (0, 255, 0), 2)
    
    # 绘制近似多边形的所有角点（红色圆圈+编号）
    for i, point in enumerate(approx):
        x, y = point[0]
        # 绘制红色圆点
        cv2.circle(frame_cut, (x, y), 5, (0, 0, 255), -1)
        # 标注点编号
        cv2.putText(frame_cut, str(i), (x-8, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    
    # 显示结果
    cv2.imshow('Contour with Corner Points', frame_cut)
    cv2.waitKey(0)


# for approx in approxs:
#     print(len(approx))
#     cv2.polylines(result, [approx.astype(np.int32)], True, (0, 0, 255), 2)

cv2.imshow('Result', result)


cv2.waitKey(0)
cv2.destroyAllWindows()