import cv2
import numpy as np

def detect_rectangle(image, target_aspect_ratio=0.66, min_area=18000, max_area=90000):
    """
    在图像中准确检测矩形
    :param image: 输入图像(BGR格式)
    :param target_aspect_ratio: 目标矩形的宽高比(可选)
    :param min_area: 最小面积阈值
    :param max_area: 最大面积阈值(可选)
    :return: 检测到的矩形角点列表(左上, 右上, 右下, 左下)
    """
    # 1. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 应用自适应阈值处理 - 更好地处理光照变化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    
    # # 3. 降噪处理
    # kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # # 4. 边缘检测 (可选，增强轮廓)
    # edges = cv2.Canny(thresh, 50, 150)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    # 6. 筛选轮廓
    rectangles = []
    for cnt in contours:
        
        # 6.1 面积筛选
        area = cv2.contourArea(cnt)
        
        if area < min_area or (max_area and area > max_area):
            print("面积不对", area)
            continue

        cv2.polylines(result, [cnt], True, (0, 255, 0), 3)
        cv2.imshow("Detected Rectangle", result)
        cv2.waitKey(0)
            
        # 6.2 多边形近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 6.3 筛选四边形
        if len(approx) == 4:
            # 6.4 凸性检查
            if not cv2.isContourConvex(approx):
                print("不具有凸性")
                continue
                
            # 6.5 角度检查 - 确保是矩形
            angles = []
            points = approx.reshape(4, 2)
            
            # 计算四个内角
            for i in range(4):
                v1 = points[(i+1) % 4] - points[i]
                v2 = points[(i-1) % 4] - points[i]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
            
            # 检查角度是否接近90度(允许±15度误差)
            if all(80 <= angle <= 100 for angle in angles):
                # 6.6 宽高比筛选(可选)
                if target_aspect_ratio:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    # 允许±10%的误差
                    if not (0.9 * target_aspect_ratio <= aspect_ratio <= 1.1 * target_aspect_ratio):
                        print("宽高比不对")
                        continue
                
                rectangles.append(points)
    
    # 7. 返回结果
    if rectangles:
        # 如果找到多个矩形，选择面积最大的
        largest_rect = max(rectangles, key=cv2.contourArea)
        return largest_rect
    
    return None

# 使用示例
if __name__ == "__main__":
    # 读取图像
    image = cv2.imread(r"project\2025\imgs\captured_images\image_11.jpg")

    
    # 检测矩形 (可选：设置目标宽高比)
    rectangle = detect_rectangle(image)
    
    if rectangle is not None:
        # 在图像上绘制检测到的矩形
        result = image.copy()
        cv2.polylines(result, [rectangle.astype(np.int32)], True, (0, 255, 0), 3)
        
        # 标记角点
        for i, point in enumerate(rectangle):
            cv2.circle(result, tuple(point.astype(np.int32)), 8, (0, 0, 255), -1)
            cv2.putText(result, str(i), tuple(point.astype(np.int32)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow("Detected Rectangle", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未检测到矩形")

"""
请你帮我检测图像中的图形，图形有可能是正方形，三角形，圆形。对于正方形和三角形，计算出边长的平均值，对于圆形，得到圆形的直径。帮我封装成一个函数，输入img，输出测量结果（边长或直径）
"""