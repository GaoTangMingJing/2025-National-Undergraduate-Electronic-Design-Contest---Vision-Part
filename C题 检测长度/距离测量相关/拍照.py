import cv2
import os
import datetime
import myutils  # 确保myutils模块已定义

# 创建保存照片的文件夹
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# 获取当前文件夹中已存在的最大编号
existing_files = [f for f in os.listdir(save_dir) if f.startswith("image_") and f.endswith(".jpg")]
if existing_files:
    max_num = max([int(f.split('_')[1].split('.')[0]) for f in existing_files])
    count = max_num + 1
else:
    count = 1

# 初始化摄像头
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture(1)

# 检查视频是否成功打开
if cap.isOpened():
    print("摄像头打开成功")
else:
    print("摄像头打开失败")

# 设置视频编解码器
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 设置MJPG

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 设置帧率
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("拍照程序已启动")
print("按 [空格键] 拍照")
print("按 [ESC] 或 [q] 退出")

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    # 翻转画面
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)

    if not ret:
        print("无法获取画面")
        break
    
    # 显示实时画面
    cv2.imshow("Camera", myutils.resize(frame, width=1000))
    
    # 获取按键输入
    key = cv2.waitKey(1)
    
    # 按ESC或q退出
    if key == 27 or key == ord('q'):
        break
    
    # 按空格键拍照
    elif key == 32:
        # 生成顺序文件名
        filename = f"{save_dir}/image_{count}.jpg"
        
        # 保存图片
        cv2.imwrite(filename, frame)
        print(f"已保存照片: {filename}")
        count += 1  # 递增计数器

# 释放资源
cap.release()
cv2.destroyAllWindows()