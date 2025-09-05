import functions

import cv2
def drawRectangle(corner, result):
        # 绘制闭合多边形
        cv2.polylines(result, [corner], isClosed=True, color=(255, 0, 0), thickness=2)
        # 标记四个角点
        for i, point in enumerate(corner.squeeze()):
            cv2.circle(result, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(result, f"{i}", tuple(point), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

frame = cv2.imread(r"project\2025\imgs\image_26.jpg")
result = frame.copy()
# cv2.imshow("frame", frame)


target = functions.detect_rectangle(frame)
drawRectangle(target, result)

target = functions.order_points(target)

x, y, w, h = cv2.boundingRect(target)
                
# 截取矩形区域
frame_cut = frame[y:y+h, x:x+w]

_, cnts, squares = functions.detect_min_square(frame_cut)
print(squares)

if squares is not None:
    i = 0
    for square in squares:
        cnt = square['cnt']

        # 截取旋转的正方形
        cropped_square = functions.get_square_num(frame_cut, cnt)

        cv2.imshow(f"Cropped Square{i}", cropped_square)

        # predict_num = predict_image(cropped_square)
        i += 1

# cv2.imshow("result", result)
# cv2.imshow("frame", frame)
cv2.imshow("frame_cut", frame_cut)
cv2.waitKey(0)
cv2.destroyAllWindows()