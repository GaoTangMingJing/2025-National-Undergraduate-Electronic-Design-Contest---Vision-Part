import cv2
import numpy as np
import random
import os

def generate_images(num_images=10, output_dir=""):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    width, height = 210, 297
    border_thickness = 20

    for img_idx in range(num_images):
        # 创建白底图像
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # 加黑色边框
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (0, 0, 0), border_thickness)

        num_squares = random.randint(1, 5)
        for _ in range(num_squares):
            side = random.randint(60, 120)
            angle_deg = random.uniform(0, 360)
            angle_rad = np.deg2rad(angle_deg)

            # 安全边界避免正方形出界
            margin = border_thickness + side // 2 + 5
            cx = random.randint(margin, width - margin)
            cy = random.randint(margin, height - margin)

            half = side / 2
            pts = np.array([
                [-half, -half],
                [ half, -half],
                [ half,  half],
                [-half,  half]
            ])

            R = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ])
            rot_pts = np.dot(pts, R) + [cx, cy]
            rot_pts = np.round(rot_pts).astype(np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(image, [rot_pts], color=(0, 0, 0))

        # 保存图像
        save_path = os.path.join(output_dir, f"image_{img_idx:04d}.png")
        cv2.imwrite(save_path, image)
        print(f"Saved: {save_path}")

# 使用示例：生成50张图像
path = r"project\2025\imgs\yolov5\no_num"
generate_images(num_images=500, output_dir=path)
