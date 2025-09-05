import cv2
import functions

data = b'\x09\x00'
print(data)  # 输出: b'\t\x00'

a = [[0,0], [1,1]]

print(a+[2,2])

# print((10.0 - 4)/(29.7 - 4))

# img = cv2.imread(r"project\2025\imgs\question4_3.png")
# print(functions.detect_rotated_rectangle(img))

# cv2.waitKey(0)
# cv2.destroyAllWindows() 

# points = [
#     [300, 120],
#     [150, 80],   # 左上角附近（假设）
#       # 右上角附近（假设）
#     [280, 250],  # 右下角附近（假设）
#     [100, 200]   # 左下角附近（假设）
# ]

# print(functions.order_points(points))