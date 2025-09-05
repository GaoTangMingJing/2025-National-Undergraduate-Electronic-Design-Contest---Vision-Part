import cv2
import functions
import numpy as np


import numpy as np
import cv2

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
        # if abs(angle - np.pi/2) < np.pi/12 and cross < 0:
        #     square_corners[i] = True
        
        if cross < 0 and abs(angle - np.pi/2) < np.pi/4:
            square_corners[i] = True
    
    for i, point in enumerate(points):
        x, y = point
        if square_corners[i]:
            # 绘制红色圆点
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
            # 标注点编号
            cv2.putText(result, str(i), (x-8, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            # 绘制蓝色圆点
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
            # 标注点编号
            cv2.putText(result, str(i), (x-8, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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
        cv2.imshow("img", result)
        cv2.waitKey(0)
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
    
    print("squares:", squares)
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

    cv2.imshow('Result', result)
    cv2.waitKey(0)
    
    return squares






frame = cv2.imread(r"project\2025\imgs\question5_3.png")

target = functions.detect_rectangle(frame)

# 计算正外接矩形
x, y, w, h = cv2.boundingRect(target)

# 截取矩形区域
frame_cut = frame[y:y+h, x:x+w]
result = frame_cut.copy()
# cv2.imwrite("binary.png", result)


gray = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2GRAY)

otsu_thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("thresh", binary_image)

contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_area = 1000
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
    
    # if len(approx) == 7:
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
    
    cv2.imshow("img", frame_cut)
    cv2.waitKey(0)

    ############################################################
    squares = process_squares(approx)
    
    for square in squares:
        points = square[0]
        points = np.array(points, dtype=int)
        
        cv2.polylines(result, [points], True, (255, 255, 0), 2)

        print("边长:", square[1])

    

    
    # 显示结果
    cv2.imshow('Contour with Corner Points', frame_cut)
    cv2.waitKey(0)


# for approx in approxs:
#     print(len(approx))
#     cv2.polylines(result, [approx.astype(np.int32)], True, (0, 0, 255), 2)

cv2.imshow('Result', result)
cv2.imshow('Result', cv2.resize(result, (600, 800)))

cv2.waitKey(0)
cv2.destroyAllWindows()