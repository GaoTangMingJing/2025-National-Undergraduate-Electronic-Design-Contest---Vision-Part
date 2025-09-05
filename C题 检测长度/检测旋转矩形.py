import cv2
import functions
import numpy as np
frame = cv2.imread(r"project\2025\imgs\question4_3.png")
frame_copy = frame.copy()

target = functions.detect_rotated_rectangle(frame)


cv2.polylines(frame_copy, [target.astype(np.int32)], True, (255, 0, 255), 2)
cv2.imshow("frame_copy", frame_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()