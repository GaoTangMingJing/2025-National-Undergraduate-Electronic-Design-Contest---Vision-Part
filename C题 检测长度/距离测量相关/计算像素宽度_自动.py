import cv2
import os
import numpy as np

image_folder = r"project\2025\imgs\captured_images3"
height_pix = []

for i in range(1, 22):  # 从 image_1 到 image_21
    filename = f"image_{i}.jpg"
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        print(f"[警告] 图片 {image_path} 不存在，跳过")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"[警告] 无法读取图片 {filename}，跳过")
        continue

    img_cut = img.copy()
    gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_quad = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 19000:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and area > max_area:
            max_area = area
            max_quad = approx

    if max_quad is not None:
        points = max_quad.reshape(4, 2)

        # 可选：绘制轮廓用于可视化
        drawing_image = img_cut.copy()
        cv2.polylines(drawing_image, [max_quad], True, (0, 255, 0), 2)

        # 计算上下边长（根据轮廓顺序）
        upper_length = np.linalg.norm(points[0] - points[1])
        lower_length = np.linalg.norm(points[2] - points[3])
        avg_length = (upper_length + lower_length) / 2
         
        height_pix.append(round(avg_length, 2))

        print(f"[{filename}] 上: {upper_length:.2f} 下: {lower_length:.2f} 平均: {avg_length:.2f}")

        # 可视化（可注释）
        cv2.imshow("Detected", drawing_image)
        cv2.waitKey(0)  # 显示 300 毫秒
    else:
        print(f"[{filename}] 未检测到有效四边形")
        height_pix.append(0.0)

cv2.destroyAllWindows()

# 输出最终结果
print("\n========== 所有图像的平均高度 ==========")
for idx, h in enumerate(height_pix, start=1):
    print(f"image_{idx}: {h:.2f} px")

print(height_pix)
