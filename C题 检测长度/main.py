import cv2
import numpy as np
import time
import myutils
import functions
import serial
import threading  # 添加线程支持
import queue     # 添加线程安全的队列
from model import predict_image

class StateType:
    FREE_STATE = 0
    CAP_INIT = 1
    BASIC_QUESTIONS = 2
    EXPANSIVE_1 = 3         # 正方形不重叠
    EXPANSIVE_2 = 4         # 重叠
    EXPANSIVE_3 = 5         # 数字识别
    EXPANSIVE_4 = 6         # 旋转

class StateMachine:
    def __init__(self):
        # 摄像头初始化
        self.cap = None
        self.initialize_camera()
        # 串口初始化
        self.ser = None
        self.initialize_serial()
        # 通信线程初始化
        self.input_queue = queue.Queue()  # 线程安全的队列
        self.msg = ""                     # 当前接收的消息
        self.running = True               # 线程运行标志
        self.input_thread1 = threading.Thread(target=self.console_input_thread)
        self.input_thread2 = threading.Thread(target=self.serial_read_thread)
        self.input_thread1.daemon = True   # 设置为守护线程
        self.input_thread1.start()
        self.input_thread2.daemon = True   # 设置为守护线程
        self.input_thread2.start()


        # 变量初始化
        self.A4_REAL_HEIGHT = 29.7 - 4
        self.A4_REAL_WIDTH = 21.0 - 4
        self.A4_pix_height = None
        self.A4_pix_width = None
        self.measure_finished = False
        self.target = None
        self.distance = None
        self.side_length = None
        self.target_front_view = None
        self.num = None


    def run(self):
        self.start()
        while True:
            # 处理所有待处理的控制台输入
            self.process_input_queue()
            
            # 运行状态机
            self.update()
            
            # 添加短暂延迟减少CPU占用
            time.sleep(0.5)
    
    def start(self):
        self.curState = StateType.FREE_STATE

    def update(self):
        match self.curState:
            case StateType.FREE_STATE:
                print("当前状态: FREE_STATE")
                self.OnFREE_STATE()
            case StateType.CAP_INIT:
                print("当前状态: CAP_INIT")
                self.OnCAP_INIT()
            case StateType.BASIC_QUESTIONS:
                self.OnBASIC_QUESTIONS()
            case StateType.EXPANSIVE_1:
                self.OnEXPANSIVE_1()
            case StateType.EXPANSIVE_2:
                self.OnEXPANSIVE_2()
            case StateType.EXPANSIVE_3:
                self.OnEXPANSIVE_3()
            case StateType.EXPANSIVE_4:
                self.OnEXPANSIVE_4()
                pass 
    
    def OnFREE_STATE(self):
        # 防止有缓存
        ret, frame = self.cap.read()
        """空闲状态处理"""
        if self.msg:
            if self.msg == "0":
                self.curState = StateType.FREE_STATE
                print("转换为状态: FREE_STATE")
            elif self.msg == "1":
                self.curState = StateType.CAP_INIT  
                print("转换为状态: CAP_INIT")
            elif self.msg == "2":
                self.curState = StateType.BASIC_QUESTIONS 
                print("转换为状态: BASIC_QUESTIONS")
            elif self.msg == "3":
                self.curState = StateType.EXPANSIVE_1 
                print("转换为状态: EXPANSIVE_1")
            elif self.msg == "4":
                self.curState = StateType.EXPANSIVE_2 
                print("转换为状态: EXPANSIVE_2")
            elif self.msg == "5":
                self.curState = StateType.EXPANSIVE_3 
                print("转换为状态: EXPANSIVE_3")
            elif self.msg == "6":
                self.curState = StateType.EXPANSIVE_4 
                print("转换为状态: EXPANSIVE_4")
            elif self.msg.lower() == 'q':
                self.cleanup()
                exit(0)
            # 清空消息
            self.msg = ""
    
    def OnCAP_INIT(self):
        frame, result = self.getFrame()
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        if self.msg:
            if self.msg == "0":
                self.curState = StateType.FREE_STATE
                print("转换为状态: FREE_STATE")
            elif self.msg == "1":
                self.curState = StateType.CAP_INIT  
                print("转换为状态: CAP_INIT")
            elif self.msg == "2":
                self.curState = StateType.BASIC_QUESTIONS 
                print("转换为状态: BASIC_QUESTIONS")
            elif self.msg == "3":
                self.curState = StateType.EXPANSIVE_1 
                print("转换为状态: EXPANSIVE_1")
            elif self.msg == "4":
                self.curState = StateType.EXPANSIVE_2 
                print("转换为状态: EXPANSIVE_2")
            elif self.msg == "5":
                self.curState = StateType.EXPANSIVE_3 
                print("转换为状态: EXPANSIVE_3")
            elif self.msg == "6":
                self.curState = StateType.EXPANSIVE_4 
                print("转换为状态: EXPANSIVE_4")
            elif self.msg.lower() == 'q':
                self.cleanup()
                exit(0)
            # 清空消息
            self.msg = ""
    
    # 基础题目
    def OnBASIC_QUESTIONS(self):
        # 像素长度测量，取10次的平均
        A4_pix_height_measurements = []
        pix_length_measurements = []
        # 初始化
        self.A4_pix_height = None
        self.measure_finished = False
        self.target = None
        self.distance = None
        self.side_length = None

        while True:
            frame, result = self.getFrame()
            # cv2.imwrite("question5_3.png", frame)

            # 检测目标
            _target = functions.detect_rectangle(frame)
            if _target is not None:
                self.target = _target
            
            # # 绘图
            # if self.target is not None:
            #     self.drawRectangle(self.target, result)

            # 计算距离
            if self.target is not None:
                # 角点排序
                self.target = functions.order_points(self.target)
                # 计算A4高度的像素值
                left_height = np.linalg.norm(self.target[0] - self.target[3])
                right_height = np.linalg.norm(self.target[1] - self.target[2])
                # 计算平均值
                A4_pix_height = (left_height + right_height) / 2

                A4_pix_height_measurements.append(A4_pix_height)
                
                # #计算距离
                # distance = functions.predict_distance(A4_pix_height)

                # cv2.putText(result, f"distance:{distance} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)


            # 识别图形
            if self.target is not None:
                # 计算正外接矩形
                x, y, w, h = cv2.boundingRect(self.target)
                
                # 截取矩形区域
                frame_cut = frame[y:y+h, x:x+w]
                
                # # 在原图上绘制外接矩形（可选，绿色框，线宽2像素）
                # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

                detected_results = functions.detect_shapes_and_measure(frame_cut)
                if detected_results is not None:
                    #像素长度
                    pix_length = detected_results.get("length")
                    pix_length_measurements.append(pix_length)

                    #实际长度
                    side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, A4_pix_height)

                    # # 画图
                    # cnt = detected_results.get("cnt")
                    # offset = np.array([x, y], dtype=np.int32)
                    # cnt = cnt + offset
                    # cv2.polylines(result, [cnt], True, (0, 255, 0), 2)

                    # cv2.putText(result, f"side_length:{side_length} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

            if len(A4_pix_height_measurements) >= 10 and len(pix_length_measurements) >= 10:
                self.measure_finished = True
                self.A4_pix_height = sum(A4_pix_height_measurements) / len(A4_pix_height_measurements)
                # 计算距离
                self.distance = functions.predict_distance(self.A4_pix_height)
                # 计算边长
                self.side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, self.A4_pix_height)
                
                print("距离:", self.distance)
                print("边长:", self.side_length)
                self.send(self.distance, self.side_length + 0.3)

            # 显示图片
            # cv2.imshow("Detection Result", result)
            # # cv2.waitKey(0)
            # cv2.waitKey(1)

            # 状态转换
            # 测量结束
            if self.measure_finished:
                self.curState = StateType.FREE_STATE
                cv2.destroyAllWindows() 
                break
            # 退出
            # 处理消息
            self.process_input_queue()
            if self.msg:
                if self.msg == "0":
                    self.curState = StateType.FREE_STATE
                    break
                if self.msg.lower() == 'q':
                    self.cleanup()
                    exit(0)
                
                # 清空消息
                self.msg = ""
    
    # 发挥部分1 不重叠矩形 3
    def OnEXPANSIVE_1(self):
        # 像素长度测量，取10次的平均
        A4_pix_height_measurements = []
        pix_length_measurements = []
        # 初始化
        self.A4_pix_height = None
        self.measure_finished = False
        self.target = None
        self.distance = None
        self.side_length = None

        while True:
            frame, result = self.getFrame()

            # 检测目标
            _target = functions.detect_rectangle(frame)
            if _target is not None:
                self.target = _target

            # # 绘图
            # if self.target is not None:
            #     self.drawRectangle(self.target, result)

            # 计算距离
            if self.target is not None:
                # 角点排序
                self.target = functions.order_points(self.target)
                # 计算A4高度的像素值
                left_height = np.linalg.norm(self.target[0] - self.target[3])
                right_height = np.linalg.norm(self.target[1] - self.target[2])
                # 计算平均值
                A4_pix_height = (left_height + right_height) / 2

                A4_pix_height_measurements.append(A4_pix_height)

                # 计算距离
                # distance = functions.predict_distance(A4_pix_height)

                # cv2.putText(result, f"distance:{distance} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

            # 识别图形
            if self.target is not None:
                # 计算正外接矩形
                x, y, w, h = cv2.boundingRect(self.target)
                
                # 截取矩形区域
                frame_cut = frame[y:y+h, x:x+w]
                
                # 在原图上绘制外接矩形（可选，绿色框，线宽2像素）
                # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # 检测最小矩形
                detected_results, cnts, _ = functions.detect_min_square(frame_cut)
                if detected_results is not None:
                    #像素长度
                    pix_length = detected_results.get("length")
                    pix_length_measurements.append(pix_length)

                    #实际长度
                    side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, A4_pix_height)

                    # # 画图
                    # cnt = detected_results.get("cnt")
                    # offset = np.array([x, y], dtype=np.int32)
                    # cnt = cnt + offset
                    # cv2.polylines(result, [cnt], True, (0, 255, 0), 2)

                    # cv2.putText(result, f"side_length:{side_length} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

                    # for cnt in cnts:
                    #     cnt = cnt + offset
                    #     self.drawRectangle(cnt, result)
            
            if len(A4_pix_height_measurements) >= 10 and len(pix_length_measurements) >= 10:
                self.measure_finished = True
                self.A4_pix_height = sum(A4_pix_height_measurements) / len(A4_pix_height_measurements)
                # 计算距离
                self.distance = functions.predict_distance(self.A4_pix_height)
                # 计算边长
                self.side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, self.A4_pix_height)
                
                print("距离:", self.distance)
                print("边长:", self.side_length)
                self.send(self.distance, self.side_length + 0.3)



            # 显示图片
            # cv2.imshow("Detection Result", result)
            # # cv2.waitKey(0)
            # cv2.waitKey(1)

            # 状态转换
            # 测量结束
            if self.measure_finished:
                self.curState = StateType.FREE_STATE
                cv2.destroyAllWindows() 
                break
            # 退出
            # 处理消息
            self.process_input_queue()
            if self.msg:
                if self.msg == "0":
                    self.curState = StateType.FREE_STATE
                    break
                if self.msg.lower() == 'q':
                    self.cleanup()
                    exit(0)
                
                # 清空消息
                self.msg = ""

    # 发挥部分2 重叠矩形 4
    def OnEXPANSIVE_2(self):
        A4_pix_height_measurements = []
        pix_length_measurements = []

        # 初始化
        self.A4_pix_height = None
        self.measure_finished = False
        self.target = None
        self.distance = None
        self.side_length = None

        while True:
            frame, result = self.getFrame()

            # 检测目标
            _target = functions.detect_rectangle(frame)
            if _target is not None:
                self.target = _target

            # # 绘图
            # if self.target is not None:
            #     self.drawRectangle(self.target, result)

            # 计算距离
            if self.target is not None:
                # 角点排序
                self.target = functions.order_points(self.target)
                # 计算A4高度的像素值
                left_height = np.linalg.norm(self.target[0] - self.target[3])
                right_height = np.linalg.norm(self.target[1] - self.target[2])
                # 计算平均值
                A4_pix_height = (left_height + right_height) / 2

                A4_pix_height_measurements.append(A4_pix_height)

                # 计算距离
                # distance = functions.predict_distance(A4_pix_height)

                # cv2.putText(result, f"distance:{distance} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

            # 识别图形
            if self.target is not None:
                # 计算正外接矩形
                x, y, w, h = cv2.boundingRect(self.target)
                
                # 截取矩形区域
                frame_cut = frame[y:y+h, x:x+w]
                
                # 在原图上绘制外接矩形（可选，绿色框，线宽2像素）
                # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)


                ##################################
                approxs = functions.before_process(frame_cut)


                squares = []
                detected_results = None
                if approxs != None:
                    for approx in approxs:
                        squares += functions.process_squares(approx)
                
                if squares != []:
                    side_lengths = [side for _, side in squares]
                    detected_results = min(side_lengths)
                    cnts = [points for points, _ in squares]
                    cnts = np.array(cnts, dtype=np.int32)

                ##################################

                # 检测重叠矩形中的最小矩形
                # detected_results, cnts = functions.process_squares(frame_cut)
                if detected_results is not None:
                    #像素长度
                    pix_length = detected_results
                    pix_length_measurements.append(pix_length)

                    #实际长度
                    side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, A4_pix_height)

                    # # 画图
                    # cv2.putText(result, f"side_length:{side_length} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

                    # offset = np.array([x, y], dtype=np.int32)
                    # for cnt in cnts:
                    #     cnt = cnt + offset
                    #     self.drawRectangle(cnt, result)

            if len(A4_pix_height_measurements) >= 10 and len(pix_length_measurements) >= 10:
                self.measure_finished = True
                self.A4_pix_height = sum(A4_pix_height_measurements) / len(A4_pix_height_measurements)
                # 计算距离
                self.distance = functions.predict_distance(self.A4_pix_height)
                # 计算边长
                self.side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, self.A4_pix_height)
                
                print("距离:", self.distance)
                print("边长:", self.side_length)
                self.send(self.distance, self.side_length + 0.3)



            # 显示图片
            # cv2.imshow("Detection Result", result)
            # # cv2.waitKey(0)
            # cv2.waitKey(1)

            # 状态转换
            # 测量结束
            if self.measure_finished:
                self.curState = StateType.FREE_STATE
                cv2.destroyAllWindows() 
                break
            # 退出
            # 处理消息
            self.process_input_queue()
            if self.msg:
                if self.msg == "0":
                    self.curState = StateType.FREE_STATE
                    break
                if self.msg.lower() == 'q':
                    self.cleanup()
                    exit(0)
                
                # 清空消息
                self.msg = ""
        
    # 发挥部分3 识别数字 5
    def OnEXPANSIVE_3(self):
        # self.num = 8

        A4_pix_height_measurements = []
        pix_length_measurements = []

        # 初始化
        self.A4_pix_height = None
        self.measure_finished = False
        self.target = None
        self.distance = None
        self.side_length = None

        while True:
            frame, result = self.getFrame()

            # 检测目标
            _target = functions.detect_rectangle(frame)
            if _target is not None:
                self.target = _target

            # 绘图
            if self.target is not None:
                self.drawRectangle(self.target, result)
                pass


            # 计算距离
            if self.target is not None:
                # 角点排序
                self.target = functions.order_points(self.target)
                # 计算A4高度的像素值
                left_height = np.linalg.norm(self.target[0] - self.target[3])
                right_height = np.linalg.norm(self.target[1] - self.target[2])
                # 计算平均值
                A4_pix_height = (left_height + right_height) / 2

                A4_pix_height_measurements.append(A4_pix_height)

                # # 计算距离
                # distance = functions.predict_distance(A4_pix_height)

                # cv2.putText(result, f"distance:{distance} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

            # 识别图形
            if self.target is not None:
                # 计算正外接矩形
                x, y, w, h = cv2.boundingRect(self.target)
                
                # 截取矩形区域
                frame_cut = frame[y:y+h, x:x+w]
                
                # 在原图上绘制外接矩形（可选，绿色框，线宽2像素）
                # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # 检测重合矩形中的数字
                # ##################################
                # approxs = functions.before_process(frame_cut)


                # squares = []
                # detected_results = None
                # if approxs != None:
                #     for approx in approxs:
                #         squares += functions.process_squares(approx)
                
                # if squares != []:
                #     # 筛选正方形：宽高比在0.9-1.1之间的视为正方形
                #     square_cnts = []
                #     for cnt, side in squares:
                #         # 计算轮廓的最小外接矩形
                #         rect = cv2.minAreaRect(np.array(cnt, dtype=np.int32))
                #         (w, h) = rect[1]  # 获取宽度和高度
                        
                #         # 确保分母不为零
                #         if min(w, h) == 0:
                #             continue
                        
                #         # 计算宽高比（总是≥1）
                #         aspect_ratio = max(w, h) / min(w, h)
                        
                #         # 保留宽高比接近1的轮廓（正方形）
                #         if aspect_ratio <= 1.2:  # 允许10%的误差
                #             square_cnts.append((cnt, side))

                #     squares = square_cnts

                #     cnts = [points for points, _ in squares]
                #     cnts = np.array(cnts, dtype=np.int32)

                #     offset = np.array([x, y], dtype=np.int32)
                #     for cnt in cnts:
                #         cnt = cnt + offset
                #         self.drawRectangle(cnt, result)
                
                # 选中矩形处理
                # ########
                # # _, cnts, squares = functions.detect_min_square(frame_cut)
                # if squares is not None:
                #     i = 0
                #     for square in squares:
                #         cnt = square['cnt']

                #         # 截取旋转的正方形
                #         cropped_square = functions.extract_inner_circle_from_rotated_square(frame_cut, cnt)

                #         cv2.imshow(f"Cropped Square{i}", cropped_square)
                #         cv2.waitKey(1)

                #         predict_num = predict_image(cropped_square)
                        

                #         # 显示或处理截取的正方形
                #         print(predict_num)
                #         # cv2.imshow(f"Cropped Square{i}", cropped_square)
                #         i += 1
                        

                #         if predict_num == self.num:
                #             #像素长度
                #             pix_length = square['length']
                #             pix_length_measurements.append(pix_length)

                #             #实际长度
                #             side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, A4_pix_height)

                            
                #             cv2.putText(result, f"side_length:{side_length} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

                #         offset = np.array([x, y], dtype=np.int32)

                #         cnt = cnt + offset
                #         self.drawRectangle(cnt, result)

                ################################

                # 检测非重合矩形中的数字
                print("num是", self.num)
                _, cnts, squares = functions.detect_min_square(frame_cut)
                # print("squares", squares)
                
                if squares is not None:
                    i = 0
                    for square in squares:
                        cnt = square['cnt']

                        # 截取旋转的正方形
                        # cropped_square = functions.extract_inner_circle_from_rotated_square(frame_cut, cnt)
                        cropped_square = functions.get_square_num(frame_cut, cnt)

                        # cv2.imshow(f"Cropped Square{i}", cropped_square)
                        # cv2.waitKey(1)

                        predict_num = predict_image(cropped_square)
                        

                        # 显示或处理截取的正方形
                        print("预测数字:", predict_num)
                        i += 1
                        

                        if predict_num == self.num:
                            #像素长度
                            pix_length = square['length']
                            pix_length_measurements.append(pix_length)


                            # #实际长度
                            # side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, A4_pix_height)
                            # # print("边长:", side_length)
                            # # self.send(0, side_length)
                            
                            # cv2.putText(result, f"side_length:{side_length} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

                        # # 画图
                        # offset = np.array([x, y], dtype=np.int32)
                        # cnt = cnt + offset
                        # self.drawRectangle(cnt, result)


            if len(A4_pix_height_measurements) >= 5 and len(pix_length_measurements) >= 5:
                self.measure_finished = True
                self.A4_pix_height = sum(A4_pix_height_measurements) / len(A4_pix_height_measurements)
                # 计算距离
                self.distance = functions.predict_distance(self.A4_pix_height)
                # 计算边长
                self.side_length = functions.get_real_length(pix_length, self.A4_REAL_HEIGHT, self.A4_pix_height)
                
                print("距离:", self.distance)
                print("边长:", self.side_length)
                self.send(self.distance, self.side_length + 0.3)


            # 显示图片
            # cv2.imshow("Detection Result", result)
            # # # cv2.waitKey(0)
            # cv2.waitKey(1)

            # 状态转换
            # 测量结束
            if self.measure_finished:
                self.curState = StateType.FREE_STATE
                cv2.destroyAllWindows() 
                break
            # 退出
            # 处理消息
            self.process_input_queue()
            if self.msg:
                if self.msg == "0":
                    self.curState = StateType.FREE_STATE
                    break
                if self.msg.lower() == 'q':
                    self.cleanup()
                    exit(0)
                
                # 清空消息
                self.msg = ""

        
    # 发挥部分4 旋转
    def OnEXPANSIVE_4(self):
        # 像素长度测量，取10次的平均
        side_length_measurements = []
        # 初始化
        self.measure_finished = False
        self.target = None
        self.side_length = None

        while True:
            frame, result = self.getFrame()

            # 检测目标
            _target = functions.detect_rotated_rectangle(frame)
            if _target is not None:
                self.target = _target

            # # 绘图
            # if self.target is not None:
            #     self.drawRectangle(self.target, result)


            # 透视变换
            if self.target is not None:
                # 角点排序
                self.target = functions.order_points(self.target)
                # 透视变换
                self.target_front_view = functions.transform_to_front_view(frame.copy(), self.target, self.A4_REAL_HEIGHT, self.A4_REAL_WIDTH)
                # cv2.imshow("self.target_front_view", self.target_front_view)
                # cv2.waitKey(1)
                # 变换后像素高度
                pix_front_height = 800

                # 边长检测
                detected_results = functions.detect_shapes_and_measure(self.target_front_view)

                if detected_results is not None:
                    #像素长度
                    pix_length = detected_results.get("length")

                    #实际长度
                    side_length = pix_length*self.A4_REAL_HEIGHT/pix_front_height
                    side_length_measurements.append(side_length)

                    # cv2.putText(result, f"side_length:{round(side_length, 2)} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

            if len(side_length_measurements) >= 10:
                self.measure_finished = True
                self.side_length = sum(side_length_measurements) / len(side_length_measurements)
                self.side_length = round(self.side_length, 2)

                print("长度:", self.side_length)
                self.send(0, self.side_length + 0.2)

            # 显示图片
            # cv2.imshow("Detection Result", result)
            # # cv2.waitKey(0)
            # cv2.waitKey(1)

            # 状态转换
            # 测量结束
            if self.measure_finished:
                self.curState = StateType.FREE_STATE
                cv2.destroyAllWindows() 
                break
            # 退出
            # 处理消息
            self.process_input_queue()
            if self.msg:
                if self.msg == "0":
                    self.curState = StateType.FREE_STATE
                    break
                if self.msg.lower() == 'q':
                    self.cleanup()
                    exit(0)
                
                # 清空消息
                self.msg = ""
        

    # 初始化摄像头
    def initialize_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if self.cap.isOpened():
            print("摄像头打开成功")
        else:
            print("摄像头打开失败")
            return
            
        # 设置视频编解码器
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # 设置帧率
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        print("帧率：", self.cap.get(cv2.CAP_PROP_FPS))

        # for _ in range(30):
        #     ret, frame = self.cap.read()

        #     if not ret:
        #         print("错误：无法从摄像头获取帧")
        #         return
        # ret, frame = self.cap.read()
        # print("模型准备加载")
        # # 模型加载
        # # predict_image(frame)
        # print("模型加载成功")

        
    # 初始化串口
    def initialize_serial(self):
        """初始化串口"""
        port = '/dev/ttyAMA0'
        baudrate = 115200
        self.ser = serial.Serial(port, baudrate)
        if self.ser.is_open:
            print("串口已打开")
    
    # 输入线程
    def console_input_thread(self):
        """独立线程：持续接收控制台输入"""
        while self.running:
            try:
                # 获取控制台输入
                data = input("请输入: ")
                # if data == "q" or "^C":
                #     exit(0)
                # 将数据放入队列
                self.input_queue.put(data)
            except EOFError:
                break  # 当遇到文件结束符时退出
    
    def serial_read_thread(self):
        """独立线程：持续从串口读取数据"""
        while self.running:
            try:
                # 读取串口数据
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
  
                    if data[0] != 0x55 or data[-1] != 0xFF:
                        print('包头包尾错误, data', data)
                    else:
                        data = data[1:-1]
                        # 将有效数据放入队列
                        self.input_queue.put(data)
                        print(f"收到有效数据包: {data}")
            
            except Exception as e:
                print(f"串口读取错误: {e}")
                # 可选：延迟后重试
                time.sleep(1)

    def process_input_queue(self):
        """处理输入队列中的所有消息"""
        if not self.input_queue.empty():
            self.msg = self.input_queue.get()
            print(f"收到新消息: {self.msg}")
            if isinstance(self.msg, str):
                pass
            elif isinstance(self.msg, bytes):
                if self.msg == b'\x01\x00':
                    self.msg = "2"
                elif self.msg == b'\x02\x00':
                    self.msg = "3"
                elif self.msg == b'\x02\x01':
                    self.msg = "4"
                #############################
                elif self.msg == b'\x03\x30':
                    self.msg = "5"
                    self.num = 0
                elif self.msg == b'\x03\x31':
                    self.msg = "5"
                    self.num = 1
                elif self.msg == b'\x03\x32':
                    self.msg = "5"
                    self.num = 2
                elif self.msg == b'\x03\x33':
                    self.msg = "5"
                    self.num = 3
                elif self.msg == b'\x03\x34':
                    self.msg = "5"
                    self.num = 4
                elif self.msg == b'\x03\x35':
                    self.msg = "5"
                    self.num = 5
                elif self.msg == b'\x03\x36':
                    self.msg = "5"
                    self.num = 6
                elif self.msg == b'\x03\x37':
                    self.msg = "5"
                    self.num = 7
                elif self.msg == b'\x03\x38':
                    self.msg = "5"
                    self.num = 8
                elif self.msg == b'\x03\x39':
                    self.msg = "5"
                    self.num = 9
                #############################
                elif self.msg == b'\x04\x00':
                    self.msg = "6"
                elif self.msg == b'\x09\x00':
                    self.msg = "0"
        else:
            self.msg = ""
    
    def send(self, deta_x, deta_y):
        """通过串口发送数据"""
        if not self.ser or not self.ser.is_open:
            return

        # 缩放并转为无符号整数
        deta_x = int(deta_x * 100)
        deta_y = int(deta_y * 100)

        # 构造完整帧: FE + X_H X_L + Y_H Y_L + 4字节保留 + FD
        frame = bytearray()
        frame.append(0xFE)
        frame.extend(deta_x.to_bytes(2, byteorder='big', signed=False))
        frame.extend(deta_y.to_bytes(2, byteorder='big', signed=False))
        frame.extend([0x00] * 4)
        frame.append(0xFD)

        self.ser.write(frame)
    

    def cleanup(self):
        self.running = False
        time.sleep(0.1)
        """清理资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.ser and self.ser.is_open:
            self.ser.close()
        cv2.destroyAllWindows()
        print("资源已释放")

    def getFrame(self):
        ret, frame = self.cap.read()
        
        if not ret:
            print("错误：无法从摄像头获取帧")
            return
            
        # 翻转画面
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        # 截取中间
        height, width = frame.shape[:2]

        # 计算截取尺寸
        crop_width = int(width * 0.5)
        crop_height = int(height * 0.7)

        # 计算起点
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2

        frame = frame[start_y:start_y+crop_height, start_x:start_x+crop_width]
        
        result = frame.copy()

        return frame, result

    def drawRectangle(self, corner, result):
        # 绘制闭合多边形
        cv2.polylines(result, [corner], isClosed=True, color=(255, 0, 0), thickness=2)
        # 标记四个角点
        for i, point in enumerate(corner.squeeze()):
            cv2.circle(result, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(result, f"{i}", tuple(point), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

if __name__ == "__main__":
    stateMachine = StateMachine()
    stateMachine.run()