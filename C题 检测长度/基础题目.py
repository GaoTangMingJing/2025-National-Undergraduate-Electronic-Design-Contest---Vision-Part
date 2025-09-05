import cv2
import numpy as np
import time
import myutils
import functions
import serial

# 设置画面尺寸
h = 150
w = 600

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# 检查视频是否成功打开
if cap.isOpened():
    print("摄像头打开成功")
else:
    print("摄像头打开失败")

# 设置视频编解码器
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 设置MJPG

# 设置摄像头分辨率（宽度和高度） 640x480 1280x720、1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 设置帧率（例如，30帧每秒）
cap.set(cv2.CAP_PROP_FPS, 60)

print("帧率：", cap.get(cv2.CAP_PROP_FPS))

# 初始化计算fps的参数
prev_time = time.time()
frame_count = 0
fps = 0

port = '/dev/ttyAMA0'
baudrate = 115200
ser = serial.Serial(port, baudrate)  # 串口初始化，根据实际情况修改串口号和波特率

# 确认串口是否打开
if ser.is_open:
    print("串口已打开")


def send(deta_x, deta_y):
    ser.write(bytes([0xFC]))

    deta_x = int(deta_x)
    deta_y = int(deta_y)
    
    # 将deta_x和deta_y转换为2字节（大端序）
    ser.write(deta_x.to_bytes(2, byteorder='big', signed=True))  # 有符号16位整数
    ser.write(deta_y.to_bytes(2, byteorder='big', signed=True))  # 有符号16位整数

    for i in range(4):
        ser.write(bytes([0x00]))

    ser.write(bytes([0xFB]))
    # print('send:', datas)
#######################
A4_REAL_HEIGHT = 29.7 - 4
A4_pix_height = None
square = None
distance = None
side_length = None
#######################

while True:
    ret, frame = cap.read()

    # 翻转画面
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)

    # 截取中间
    height, width = frame.shape[:2]

    # 设置截取比例（0.0-1.0）
    crop_ratio = 0.5  # 截取50%的中心区域

    # 计算截取尺寸
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio * 1.4)

    # 计算起点
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    # # 在原始图像上绘制绿色矩形框表示截取区域
    # cv2.rectangle(frame, 
    #              (start_x, start_y), 
    #              (start_x + crop_width, start_y + crop_height),
    #              (0, 255, 0),  # 绿色 (BGR格式)
    #              2)  # 线宽为2像素

    # # 显示带有标记的原始图像
    # cv2.imshow("Camera with Crop Area", frame)

    # 执行截取
    frame = frame[start_y:start_y+crop_height, start_x:start_x+crop_width]

    # 如果读取帧成功
    if not ret:
        print("错误：无法从摄像头获取帧")
        break
##########################################################################################################################
    result = frame.copy()

    _square = functions.detect_rectangle(frame)
    if _square is not None:
        square = _square
    
    # 在图像上绘制检测到的矩形
    if square is not None:
        cv2.polylines(result, [square.astype(np.int32)], True, (0, 255, 0), 2)
        
        # 标记角点
        for i, point in enumerate(square):
            cv2.circle(result, tuple(point.astype(np.int32)), 8, (0, 0, 255), -1)
            cv2.putText(result, str(i), tuple(point.astype(np.int32)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    #计算距离
    if square is not None:
        # 角点排序
        square = functions.order_points(square)
        # 计算高度像素值
        left_height = np.linalg.norm(square[0] - square[3])
        right_height = np.linalg.norm(square[1] - square[2])
        # 计算平均值
        A4_pix_height = (left_height + right_height) / 2
        
        #计算距离
        distance = functions.predict_distance(A4_pix_height)

        cv2.putText(result, f"distance:{distance} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

    #识别图形
    if square is not None:
        # 计算正外接矩形
        x, y, w, h = cv2.boundingRect(square)
        
        # 截取矩形区域
        frame_cut = frame[y:y+h, x:x+w]
        
        # 在原图上绘制外接矩形（可选，绿色框，线宽2像素）
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

        detected_results = functions.detect_shapes_and_measure(frame_cut)
        if detected_results is not None:
            pix_length = detected_results.get("length")
            
            # 画图
            cnt = detected_results.get("cnt")
            offset = np.array([x, y], dtype=np.int32)
            cnt = cnt + offset
            cv2.polylines(result, [cnt], True, (0, 255, 0), 2)

            sild_length = functions.get_real_length(pix_length, A4_REAL_HEIGHT, A4_pix_height)

            cv2.putText(result, f"sild_length:{sild_length} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
        
       

    # 显示结果
    cv2.imshow("Detected Rectangle", result)


##########################################################################################################################
    # 计算经过的时间
    current_time = time.time()
    elapsed_time = current_time - prev_time
    frame_count += 1

    # 每秒更新帧率
    if elapsed_time >= 1.0:
        fps = int(frame_count / elapsed_time)
        prev_time = current_time
        frame_count = 0

    # 显示帧率
    cv2.putText(frame, f"FPS:{fps}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20, cv2.LINE_AA)
    # cv2.putText(frame, f"FPS:{fps}", (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 20, cv2.LINE_AA)

    # 显示图像
    # cv2.imshow('frame', myutils.resize(frame, width=w))
    # cv2.imshow('frame', frame)

    # 按下任意键退出
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()