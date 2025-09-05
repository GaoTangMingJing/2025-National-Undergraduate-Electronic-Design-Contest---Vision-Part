import cv2
import os
import functions
import numpy as np

# 全局变量
drawing = False  # 是否正在绘制
ix, iy = -1, -1  # 矩形起点
fx, fy = -1, -1  # 矩形终点

# 鼠标回调函数
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y

# 读取图片
image_path = r"project\2025\imgs\captured_images3\image_21.jpg"  # 替换为你的图片路径
if not os.path.exists(image_path):
    print(f"错误：图片文件 {image_path} 不存在")
    exit()

img = cv2.imread(image_path)
if img is None:
    print("错误：无法读取图片")
    exit()

# 创建窗口并设置鼠标回调
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

print("操作说明:")
print("1. 鼠标拖拽绘制矩形区域")
print("2. 按空格键保存选中区域")
print("3. 按ESC退出")

img_cut = None
while True:
    img_copy = img.copy()
    
    # 绘制当前矩形
    if ix != -1 and iy != -1 and fx != -1 and fy != -1:
        cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 2)
    
    cv2.imshow("Image", img_copy)
    
    key = cv2.waitKey(1)
    
    if key == 27:  # ESC退出
        break
    elif key == 32:  # 空格键保存
        if ix != -1 and iy != -1 and fx != -1 and fy != -1:
            # 确保坐标有效
            x1, x2 = sorted([ix, fx])
            y1, y2 = sorted([iy, fy])
            
            # 裁剪区域
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                # cv2.imwrite("cropped_region.jpg", roi)
                img_cut = roi
                break
                # print("已保存裁剪区域到 cropped_region.jpg")
            else:
                print("错误：无效的矩形区域")
        else:
            print("请先绘制矩形区域")
# cv2.imshow("Roi", img_cut)
# cv2.waitKey(0)
cv2.destroyAllWindows()


# 转换为灰度图
gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)

# 二值化处理
otsu_thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# adjusted_thresh = otsu_thresh + otsu_offset
# _, thresh = cv2.threshold(gray, adjusted_thresh, 255, cv2.THRESH_BINARY_INV)
# thresh = cv2.bitwise_not(binary_image)
cv2.imshow("thresh", binary_image)
# cv2.waitKey(0)
# # 形态学操作（去除噪点，连接断线）
# kernel = np.ones((5, 5), np.uint8)
# processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # 按轮廓面积从大到小排序
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

drawing_image = img_cut.copy()
# 筛选符合条件的轮廓
detected_frame = None
for cnt in contours:
    # 计算轮廓面积
    area = cv2.contourArea(cnt)
    
    # 忽略太小的区域
    if area < 19000:
        continue

    print(area)
        
    # 多边形近似
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # 检测四边形（不再限制宽高比）
    if len(approx) == 4:
        # 获取四个角点
        points = approx.reshape(4, 2)
        detected_frame = points

        # 方法2：使用polylines绘制轮廓（效果相同）
        cv2.polylines(drawing_image, [approx], True, (0, 255, 0), 1)
        break

print(detected_frame)

# 计算上边长度（point 0 -> point 3）
upper_length = np.linalg.norm(points[0] - points[3])

# 计算下边长度（point 1 -> point 2）
lower_length = np.linalg.norm(points[1] - points[2])

# 计算平均值
avg_length = (upper_length + lower_length) / 2

print(f"上边长度: {upper_length:.2f} 像素")
print(f"下边长度: {lower_length:.2f} 像素")
print(f"平均长度: {avg_length:.2f} 像素")

# 4 1 2 3
# 计算上边长度（point 0 -> point 3）
upper_length = np.linalg.norm(points[0] - points[1])

# 计算下边长度（point 1 -> point 2）
lower_length = np.linalg.norm(points[3] - points[2])

# 计算平均值
avg_length = (upper_length + lower_length) / 2

print(f"上边长度: {upper_length:.2f} 像素")
print(f"下边长度: {lower_length:.2f} 像素")
print(f"平均长度: {avg_length:.2f} 像素")

cv2.imshow("Detected Quadrilateral", drawing_image)
cv2.waitKey(0)
cv2.destroyAllWindows()