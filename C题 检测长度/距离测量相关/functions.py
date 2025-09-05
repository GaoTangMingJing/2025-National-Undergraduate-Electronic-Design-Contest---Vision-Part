import cv2
import numpy as np
from scipy import interpolate
import math

def detect_rectangle(image, target_aspect_ratio=0.66, min_area=15000, max_area=100000):
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
    # cv2.imshow("thresh", thresh)
    # cv2.imwrite("thresh.png", thresh)
    # cv2.waitKey(0)
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. 筛选轮廓
    rectangles = []
    for cnt in contours:
        # 6.1 面积筛选
        area = cv2.contourArea(cnt)
        if area < min_area or (max_area and area > max_area):
            continue

        
        cv2.polylines(image, [cnt.astype(np.int32)], True, (0, 255, 0), 2)
        # cv2.imshow("cnt", image)
        # print(area)
        
        # cv2.waitKey(0)
            
        # 6.2 多边形近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 6.3 筛选四边形
        if len(approx) == 4:
            # 6.4 凸性检查
            if not cv2.isContourConvex(approx):
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
                        continue
                
                rectangles.append(points.astype(np.int32))
    
    # 7. 返回结果
    if rectangles:
        # 如果找到多个矩形，选择面积小的
        largest_rect = min(rectangles, key=cv2.contourArea)
        return largest_rect
    
    return None

def detect_rotated_rectangle(image, target_aspect_ratio=0.66, min_area=6700, max_area=100000):
    """
    在图像中准确检测绕竖直轴旋转后的矩形，即梯形，增加一个判断有一组对边平行
    :param image: 输入图像(BGR格式)
    :param min_area: 最小面积阈值
    :param max_area: 最大面积阈值(可选)
    :return: 检测到的矩形角点列表(左上, 右上, 右下, 左下)
    """
    # 1. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 应用自适应阈值处理 - 更好地处理光照变化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("thresh", thresh)
    # cv2.imwrite("thresh.png", thresh)
    # cv2.waitKey(0)
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. 筛选轮廓
    rectangles = []
    for cnt in contours:
        # 6.1 面积筛选
        area = cv2.contourArea(cnt)
        if area < min_area or (max_area and area > max_area):
            continue

        
        # cv2.polylines(image, [cnt.astype(np.int32)], True, (0, 255, 0), 2)
        # cv2.imshow("cnt", image)
        # print(area)
        # cv2.waitKey(0)
            
        # 6.2 多边形近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 6.3 筛选四边形
        if len(approx) == 4:
            # 6.4 凸性检查
            if not cv2.isContourConvex(approx):
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
            
            # 检查内角角度
            if not all(75 <= angle <= 105 for angle in angles):
                continue

            # 6.6是否有一组边平行，且一组边不平行
            # 检查是否存在邻角互补（和为180°）来判断平行边
            has_parallel_sides = False
            angle_threshold = 5  # 角度和与180°的允许偏差

            # 检查相邻角对
            for i in range(4):
                # 计算两个相邻角的和
                angle_sum = angles[i] + angles[(i+1) % 4]
                
                # 检查是否接近180°（考虑浮点误差）
                if abs(angle_sum - 180) < angle_threshold:
                    has_parallel_sides = True
                    break
            
            # 如果没有发现邻角互补，跳过该四边形
            if not has_parallel_sides:
                continue
            
            # 6.7宽高比
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            # 允许±10%的误差
            if not aspect_ratio < target_aspect_ratio:
                continue
            
            cv2.polylines(image, [cnt.astype(np.int32)], True, (0, 0, 255), 2)
            # cv2.imshow("result", image)
            # print("area", area)
            # print("angles", angles)
            # cv2.waitKey(0)
            
            rectangles.append(points.astype(np.int32))
    
    # 7. 返回结果
    if rectangles:
        # 如果找到多个矩形，选择面积小的
        rect = min(rectangles, key=cv2.contourArea)
        return rect
    
    return None

# 辅助函数：判断两条边是否平行
def are_edges_parallel(edge1, edge2, angle_threshold=5.0):
    """
    判断两条边是否平行
    :param edge1: 第一条边的向量 (dx1, dy1)
    :param edge2: 第二条边的向量 (dx2, dy2)
    :param angle_threshold: 角度阈值(度)
    :return: 是否平行
    """
    # 计算两条边的角度(弧度)
    angle1 = np.arctan2(edge1[1], edge1[0])
    angle2 = np.arctan2(edge2[1], edge2[0])
    
    # 计算角度差(考虑180度对称性)
    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, np.pi * 2 - angle_diff)
    
    # 转换为角度
    angle_diff_deg = np.degrees(angle_diff)
    
    # 检查是否平行(直接平行或180度反向平行)
    return angle_diff_deg < angle_threshold

# 凸四边形角点排序
def order_points(points):
    """
    将四边形顶点排序，x+y最小的点为第一个点，顺时针方向
    :param points: 四个点的列表，格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    :return: 排序后的点列表，x+y最小点开始顺时针方向
    """
    # 1. 找到x+y最小的点作为起点
    start_point = min(points, key=lambda p: p[0] + p[1])
    other_points = [p for p in points if not np.array_equal(p, start_point)]
    
    # 2. 计算极角（顺时针方向）
    def calculate_angle(reference, point):
        dx = point[0] - reference[0]
        dy = point[1] - reference[1]
        # 返回角度（顺时针方向，从正x轴开始）
        return math.atan2(dy, dx)  # 注意这里是dy, dx
    
    # 3. 按角度从大到小排序（顺时针方向）
    other_points_sorted = sorted(other_points,
                            key=lambda p: calculate_angle(start_point, p))
    
    # 4. 组合结果
    return np.array([start_point] + other_points_sorted).astype(np.int32)

def predict_distance(height_px):
    """
    根据像素高度预测实际距离（厘米）
    
    参数:
        height_px (float): 检测到的目标像素高度值
    
    返回:
        float: 预测的实际距离（单位：厘米）
    
    说明:
        基于校准数据：
        距离(cm): [200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
        高度(px): [172.5, 181.5, 193.04, 204.14, 217.15, 232.29, 250.09, 270.06, 292.55, 320.54, 354.02]
    """
    # 校准数据
    distances = np.array([200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100])
    height_pix = np.array([172.5, 181.5, 193.04, 204.14, 217.15, 232.29, 250.09, 270.06, 292.55, 320.54, 354.02])
    
    # 创建线性插值器（使用您提供的校准数据）
    interpolator = interpolate.interp1d(
        height_pix,
        distances,
        kind='linear',          # 线性插值
        fill_value='extrapolate' # 允许外推超出数据范围
    )
    
    # 预测并返回距离
    return round(float(interpolator(height_px)), 2)  # 四舍五入保留2位小数

def detect_shapes_and_measure(img):
    """
    检测图像中的图形（正方形、三角形、圆形），
    并返回其边长（均值）或直径的测量结果。

    参数:
        img: 输入图像（BGR）

    返回:
        results: 列表，每个元素为字典，例如：
            [{'shape': 'triangle', 'length': 52.3},
            {'shape': 'circle', 'diameter': 39.2},
            {'shape': 'square', 'length': 60.1}]
    """
    result = None

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化 + 反转
    otsu_thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("thresh", binary_image)
    # cv2.waitKey(0)

    # 轮廓检测
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 面积过滤，避免噪声
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        # 近似轮廓（保留原始轮廓则注释掉下面两行）
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # cv2.polylines(img, [cnt], True, (0, 255, 0), 3)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print(len(approx))

        if len(approx) == 3:  # 三角形
            # 计算边长
            side_lengths = []
            for i in range(3):
                pt1 = approx[i][0]
                pt2 = approx[(i+1)%3][0]
                length = np.linalg.norm(pt1 - pt2)
                side_lengths.append(length)
            
            avg_length = np.mean(side_lengths)
            std_dev = np.std(side_lengths)  # 标准差衡量边长一致性
            
            # 判断是否为正三角形（边长差异<5%且角度接近60°）
            if std_dev / avg_length < 0.1:
                result = {
                    'shape': 'triangle',
                    'length': avg_length,
                    'cnt': approx
                }

        elif len(approx) == 4:  # 四边形
            # 计算边长
            pts = approx.reshape(-1, 2)
            side_lengths = []
            # angles = []
            for i in range(4):
                pt1 = pts[i]
                pt2 = pts[(i+1)%4]
                pt3 = pts[(i+2)%4]
                
                # 边长
                length = np.linalg.norm(pt1 - pt2)
                side_lengths.append(length)
                
            
            avg_length = np.mean(side_lengths)
            length_std = np.std(side_lengths)
            
            # 判断是否为正方形（边长差异<5%且角度接近90°）
            if length_std / avg_length < 0.1 :
                result = {
                    'shape': 'square',
                    'length': avg_length,
                    'cnt': approx,
                }

        elif len(approx) > 6:  # 圆形
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            area_circle = np.pi * radius**2
            circularity = area / area_circle  # 实际面积/理想圆面积
            
            # 判断是否为正圆（圆形度>0.9）
            if circularity > 0.9:
                result = {
                    'shape': 'circle',
                    'length': 2 * radius,
                    'cnt': approx
                }

    return result

# 检测最小正方形
def detect_min_square(img):
    min_square = None
    min_area = float('inf')  # 初始化为无穷大
    squares = []  # 存储所有检测到的正方形

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化 + 反转
    otsu_thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 轮廓检测
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = []
    for cnt in contours:
        # 面积过滤，避免噪声
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        # 近似轮廓
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:  # 四边形
            cnts.append(approx)
            # 计算边长
            pts = approx.reshape(-1, 2)
            side_lengths = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
            avg_length = np.mean(side_lengths)
            length_std = np.std(side_lengths)
            
            # 判断是否为正方形（边长差异<10%）
            if length_std / avg_length < 0.1:
                # 计算面积（使用轮廓面积或实际计算）
                contour_area = cv2.contourArea(approx)

                # 创建正方形信息字典
                square_info = {
                    'shape': 'square',
                    'length': avg_length,
                    'area': contour_area,
                    'cnt': approx,
                }
                squares.append(square_info)
                
                # 更新最小正方形
                if contour_area < min_area:
                    min_area = contour_area
                    min_square = {
                        'shape': 'square',
                        'length': avg_length,
                        'area': contour_area,
                        'cnt': approx,
                    }

    return min_square, cnts, squares  # 返回最小的正方形，如果没有则返回None


def get_real_length(pix_length, A4_REAL_HEIGHT, A4_pix_height):
    real_length =  pix_length * A4_REAL_HEIGHT / A4_pix_height
    return round(real_length, 2)

def transform_to_front_view(frame, target_points, real_height, real_width):
    """
    将旋转后的矩形（梯形）透视变换回正视图
    :param frame: 原始图像
    :param target_points: 检测到的矩形角点（左上、右上、右下、左下）
    :param real_height: 矩形原始高度
    :param real_width: 矩形原始宽度
    :return: 变换后的正视图图像
    """
    # 确保输入点顺序正确（左上、右上、右下、左下）
    
    # 将点转换为浮点数格式
    src_pts = np.array(target_points, dtype=np.float32)
    # pix = 5
    # src_pts[0][0] -= pix
    # src_pts[0][1] -= pix

    # src_pts[1][0] += pix
    # src_pts[1][1] -= pix

    # src_pts[2][0] += pix
    # src_pts[2][1] += pix

    # src_pts[3][0] -= pix
    # src_pts[3][1] += pix
    
    # 计算目标矩形的尺寸（保持原始宽高比）
    aspect_ratio = real_width / real_height
    
    # 设置目标高度为原始高度的比例（可调整）
    target_height = 800  # 可根据需要调整这个值
    target_width = int(target_height * aspect_ratio)
    
    # 定义目标矩形的四个点（左上、右上、右下、左下）
    dst_pts = np.array([
        [0, 0],                          # 左上
        [target_width - 1, 0],            # 右上
        [target_width - 1, target_height - 1],  # 右下
        [0, target_height - 1]            # 左下
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # 应用透视变换
    warped = cv2.warpPerspective(frame, M, (target_width, target_height))
    
    return warped


# 检测重叠矩形
# 判断两向量是否平行
def is_parallel(v1, v2, angle_threshold=5):
    def angle_from_vector(v):
        if np.linalg.norm(v) < 1e-6:
            return 0
        dx, dy = v
        return np.arctan2(dy, dx) * 180 / np.pi
    
    a1 = angle_from_vector(v1)
    a2 = angle_from_vector(v2)
    diff = abs(a1 - a2) % 180
    return diff < angle_threshold or abs(diff - 180) < angle_threshold

# 计算向量夹角（弧度）
def vector_angle(v):
    return np.arctan2(v[1], v[0])

# 计算两直线交点
def line_intersection(line1, line2):
    p1, p2 = line1
    p3, p4 = line2
    
    A = np.array([p2[0]-p1[0], p3[0]-p4[0]], 
                 [[p2[1]-p1[1], p3[1]-p4[1]]])
    b = np.array([p3[0]-p1[0], p3[1]-p1[1]])
    
    try:
        t = np.linalg.solve(A, b)
        return p1 + t[0]*(p2-p1)
    except:
        return None

# 主处理函数
def process_squares(approx):
    # 转换为点列表并确保顺时针
    points = np.squeeze(approx).tolist()
    # if cv2.contourArea(approx) > 0:  # 如果逆时针则反转
    #     points = points[::-1]
    
    # 步骤1: 构建线段
    n = len(points)
    print("点数:", n)
    # print(points)

    lines = []
    for i in range(n):
        p1 = points[i]
        p2 = points[(i+1) % n]
        lines.append((p1, p2))
    
    # 步骤2: 过滤共线点
    changed = True
    while changed and len(points) > 4:
        changed = False
        new_points = []
        new_lines = []
        n = len(points)
        
        for i in range(n):
            prev = points[(i-1) % n]
            curr = points[i]
            next_pt = points[(i+1) % n]
            
            v1 = [curr[0]-prev[0], curr[1]-prev[1]]
            v2 = [next_pt[0]-curr[0], next_pt[1]-curr[1]]
            
            if is_parallel(v1, v2):
                changed = True
            else:
                new_points.append(curr)
        
        if changed:
            points = new_points
            n = len(points)
            lines = []
            for i in range(n):
                p1 = points[i]
                p2 = points[(i+1) % n]
                lines.append((p1, p2))
            
            print("删去了直线上的点")
            print("点数:", n)
    
    # 步骤3: 识别正方形角点
    n = len(points)
    square_corners = [False] * n
    angles = []
    
    for i in range(n):
        prev = points[(i-1) % n]
        curr = points[i]
        next_pt = points[(i+1) % n]
        
        v1 = [prev[0]-curr[0], prev[1]-curr[1]]  # 入向量
        v2 = [next_pt[0]-curr[0], next_pt[1]-curr[1]]  # 出向量
        
        # 计算叉积和角度
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-5))
        
        # 判断是否为正方形角点 (90°±15° 且 方向向下/凸)
        if cross < 0 and abs(angle - np.pi/2) < np.pi/4:
            square_corners[i] = True
    
    # for i, point in enumerate(points):
    #     x, y = point
    #     if square_corners[i]:
    #         # 绘制红色圆点
    #         cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
    #         # 标注点编号
    #         cv2.putText(result, str(i), (x-8, y), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #     else:
    #         # 绘制蓝色圆点
    #         cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
    #         # 标注点编号
    #         cv2.putText(result, str(i), (x-8, y), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 步骤4: 角点分组
    groups = []
    n = len(points)

    # 四个角点，直接返回
    if n == 4:
        square = np.array(points)
        side1 = np.linalg.norm(square[0] - square[1])  # AB
        side2 = np.linalg.norm(square[1] - square[2])  # BC
        side3 = np.linalg.norm(square[2] - square[3])  # CD
        side4 = np.linalg.norm(square[3] - square[0])  # DA
        
        # 计算平均长度
        side_length = (side1 + side2 + side3 + side4) / 4

        return [(points, side_length)]
    
    # 多个角点
    # 角点分类
    # 步骤1: 找到第一个非正方形角点的起始索引
    start_index = 0
    
    while start_index < n and square_corners[start_index]:
        start_index += 1

    # 如果全是角点 识别错误
    if start_index == n:
        print("开始角点:", start_index)
        print("是否是角点:", square_corners)
        return None
    # print("开始角点:", start_index)
    # print("是否是角点:", square_corners)

    start_idx = -1  # 初始化
    # 步骤2: 从第一个非角点开始循环
    for i in range(start_index, start_index + n + 1):
        i = i % n
        
        if square_corners[i]:
            if start_idx == -1:
                start_idx = i  # 序列起始位置
                length = 1
            else:
                length += 1
        else:
            if start_idx != -1:
                if length >= 2:
                    groups.append(('continuous', start_idx, length))
                start_idx = -1
                length = 0

    
    # 处理单个角点
    # 第一步：收集所有已被连续序列覆盖的索引
    covered_indices = set()
    for group in groups:
        if group[0] == 'continuous':
            start, length = group[1], group[2]
            # 添加连续序列中的所有索引
            for j in range(length):
                idx = (start + j) % n
                covered_indices.add(idx)
    # print("covered_indices", covered_indices)

    # 第二步：处理单个角点
    for i in range(n):
        # 只处理未被覆盖的角点
        if square_corners[i] and i not in covered_indices:
            groups.append(('single', i, 1))
    
    print("分组:", groups)
    
    # 步骤5: 处理分组并补全正方形
    squares = []

    # 寻找两线角点
    def compute_intersection(A, dir1, C, dir2):
        A = np.array(A, dtype=np.float32)
        C = np.array(C, dtype=np.float32)
        dir1 = np.array(dir1, dtype=np.float32)
        dir2 = np.array(dir2, dtype=np.float32)

        # 构造线性方程组：A + s1 * dir1 = C + s2 * dir2
        # => s1 * dir1 - s2 * dir2 = (C - A)
        M = np.column_stack((dir1, -dir2))  # shape (2, 2)
        b = C - A

        try:
            s = np.linalg.solve(M, b)
            D = A + s[0] * dir1
            return D.tolist()
        except np.linalg.LinAlgError:
            # 方向向量平行，退化为估算中点
            return ((A + C) / 2).tolist()
    
    # 处理连续3个角点
    for g in groups[:]:
        if g[0] == 'continuous' and g[2] == 3:
            start, length = g[1], g[2]
            idxs = [start, (start+1)%n, (start+2)%n]
            A, B, C = [points[i] for i in idxs]
            
            # 计算第四个点
            AB = [B[0]-A[0], B[1]-A[1]]
            BC = [C[0]-B[0], C[1]-B[1]]
            
            # 旋转90度得到垂直向量
            perp_AB = [-AB[1], AB[0]]
            perp_BC = [-BC[1], BC[0]]
            
            # 构建两条直线，过A沿 perp_AB，过C沿 perp_BC，求交点D
            D = compute_intersection(A, perp_AB, C, perp_BC)
            
            # 计算边长
            side1 = np.linalg.norm(AB)
            side2 = np.linalg.norm(BC)
            avg_side = (side1 + side2) / 2
            
            squares.append(([A, B, C, D], avg_side))
            groups.remove(g)
    
    # 处理连续2个角点
    for g in groups[:]:
        if g[0] == 'continuous' and g[2] == 2:
            start, length = g[1], g[2]
            idxs = [start, (start+1)%n]
            A, B = [points[i] for i in idxs]

            
            # 计算边长和垂直向量
            AB = [B[0]-A[0], B[1]-A[1]]
            side = np.linalg.norm(AB)
            perp = [-AB[1]/side, AB[0]/side]  # 单位垂直向量
            
            # 确定垂直方向 (检查相邻点)
            prev_idx = (start-1) % n
            next_idx = (start+2) % n
            prev_pt = points[prev_idx]
            next_pt = points[next_idx]
            
            # 选择正确的垂直方向
            vec_to_prev = [prev_pt[0]-A[0], prev_pt[1]-A[1]]
            if np.dot(perp, vec_to_prev) < 0:
                perp = [AB[1]/side, -AB[0]/side]  # 反向
            
            # 计算另外两个点
            D = [A[0] + perp[0]*side, A[1] + perp[1]*side]
            C = [B[0] + perp[0]*side, B[1] + perp[1]*side]
            
            squares.append(([A, B, C, D], side))
            groups.remove(g)
    
    
    # 处理单个角点（对角点补正方形）
    single_indices = [g[1] for g in groups if g[0] == 'single']

    if len(single_indices) == 2:
        idx1, idx2 = single_indices
        A = np.array(points[idx1], dtype=np.float32)
        C = np.array(points[idx2], dtype=np.float32)

        # 对角线中点
        M = (A + C) / 2
        v = C - A

        # 垂直方向向量（顺时针90°）
        perp = np.array([v[1], -v[0]], dtype=np.float32)

        
        # 单位化并缩放为与 v 等长的一半
        diag_length = np.linalg.norm(v)
        perp = perp / np.linalg.norm(perp) * (diag_length / 2)

        # 求 B, D 点
        B = M + perp
        D = M - perp

        square_pts = [A.tolist(), B.tolist(), C.tolist(), D.tolist()]
        squares.append((square_pts, diag_length / np.sqrt(2)))

        # 移除这两个 group，避免重复处理
        groups = [g for g in groups if g[1] not in single_indices]

    elif len(single_indices) > 2:
        pass  # 3个以上暂不处理

    # print("squares:", squares)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    
    return squares

# 主处理前处理
def before_process(frame_cut, precision=0.01):
    # precision 近似轮廓 0.005 - 0.015
    gray = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2GRAY)

    otsu_thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("thresh", binary_image)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1000
    approxs = []

    for cnt in contours:
        # 6.1 面积筛选
        area = cv2.contourArea(cnt)
        # print(f"轮廓面积: {area}")
        if area < min_area:
            continue

        # 近似轮廓 0.005 - 0.015
        epsilon = precision * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approxs.append(approx)
        # print(f"点数: {len(approx)}")
        
        # 绘制近似轮廓（红色）
        # cv2.polylines(frame_cut, [approx.astype(np.int32)], True, (0, 255, 0), 2)
        
        # # 绘制近似多边形的所有角点（红色圆圈+编号）
        # for i, point in enumerate(approx):
        #     x, y = point[0]
        #     # 绘制红色圆点
        #     cv2.circle(frame_cut, (x, y), 5, (0, 0, 255), -1)
        #     # 标注点编号
        #     cv2.putText(frame_cut, str(i), (x-8, y), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    return approxs

# 判断是否为贴边矩形
def is_near_border(approx, w, h, margin=10, threshold_ratio=0.75):
    """判断是否为贴边矩形"""
    near_count = 0
    for pt in approx:
        x, y = pt[0]
        if x <= margin or x >= w - margin or y <= margin or y >= h - margin:
            near_count += 1
    return near_count / len(approx) >= threshold_ratio


# 从图像中截取旋转的正方形区域
def crop_rotated_square(image, square_cnt):
    """
    从图像中截取旋转的正方形区域
    
    参数:
        image: 原始图像 (BGR格式)
        square_cnt: 正方形的轮廓点 (4个点)
    
    返回:
        cropped: 截取并校正后的正方形图像
    """
    # 将轮廓点转换为numpy数组
    pts = square_cnt.reshape(4, 2).astype(np.float32)
    
    # 计算旋转矩形
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    
    # 调整角度（确保宽度大于高度）
    if size[0] < size[1]:
        size = (size[1], size[0])
        angle += 90
    
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转整个图像
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), 
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # 计算旋转后正方形的位置
    rect_points = cv2.boxPoints((center, size, angle))
    rect_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]
    
    # 计算旋转后正方形的边界
    x, y, w, h = cv2.boundingRect(rect_points.astype(np.int32))
    
    # 截取正方形区域
    cropped = rotated[y:y+h, x:x+w]
    
    return cropped

def extract_inner_circle_from_rotated_square(frame_cut, cnt, output_size=96):
    """
    从旋转正方形轮廓 cnt 中提取其内接圆区域，补成正方形图像（无旋转），灰度二值化处理。
    """
    try:
        # 确保 cnt 是 [4,1,2] 或 [4,2] 格式
        cnt = np.squeeze(cnt)  # → [4, 2]
        if cnt.shape != (4, 2):
            raise ValueError("cnt 应该是 4 个角点的轮廓")

        # 转为灰度并二值化
        gray = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算中心点
        center = np.mean(cnt, axis=0).astype(int)
        cx, cy = center

        # 估算边长与半径
        dists = [np.linalg.norm(cnt[i] - cnt[(i + 1) % 4]) for i in range(4)]
        side_length = int(np.mean(dists))
        radius = side_length // 2

        # 创建黑底结果图和遮罩
        result = np.zeros((side_length, side_length), dtype=np.uint8)
        mask = np.zeros((side_length, side_length), dtype=np.uint8)
        cv2.circle(mask, (radius, radius), radius, 255, -1)

        # 计算原图裁剪区域和 result 的插入位置
        y_min = max(0, cy - radius)
        y_max = min(bin_img.shape[0], cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(bin_img.shape[1], cx + radius)

        result_y_min = max(0, radius - (cy - y_min))
        result_y_max = result_y_min + (y_max - y_min)
        result_x_min = max(0, radius - (cx - x_min))
        result_x_max = result_x_min + (x_max - x_min)

        # 裁剪并粘贴
        if y_max > y_min and x_max > x_min:
            cropped = bin_img[y_min:y_max, x_min:x_max]
            result[result_y_min:result_y_max, result_x_min:result_x_max] = cropped

        # 应用圆形遮罩
        result = cv2.bitwise_and(result, result, mask=mask)

        # 调整输出尺寸
        result = cv2.resize(result, (output_size, output_size))

        # 如果你需要RGB图像用于后续处理或模型预测，可解注释以下一行：
        # result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

    except Exception as e:
        print(f"提取内接圆时出错: {str(e)}")
        return np.zeros((output_size, output_size), dtype=np.uint8)




