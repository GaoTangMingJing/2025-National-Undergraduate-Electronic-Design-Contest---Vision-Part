import cv2

img = cv2.imread(r"project\2025\imgs\yolov5\no_num\image_000.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

otsu_thresh, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow("thresh", binary_image)

# 查找轮廓（只检测外部轮廓）
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历所有轮廓
for i, cnt in enumerate(contours):
    # 获取轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(cnt)
    
    print(f"白色区域 {i+1}: 宽度={w}px, 高度={h}px")
    
    # 可视化标记（可选）
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()