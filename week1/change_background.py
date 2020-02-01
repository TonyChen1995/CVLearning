import cv2
import numpy as np
import copy
"""
"""
img=cv2.imread('cat.jfif')
#缩放
print(cv2.cvtColor(np.uint8([[[255, 0, 0 ]]]) ,cv2.COLOR_BGR2HSV))
rows,cols,channels = img.shape
img=cv2.resize(img, None, fx=0.5, fy=0.5)
rows,cols,channels = img.shape
cv2.imshow('ori_img',img)

#转换hsv
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_blue=np.array([94, 0, 0]) # 需要多次尝试调整
upper_blue=np.array([130, 255, 255]) # 需要多次尝试调整
mask = cv2.inRange(hsv, lower_blue, upper_blue)

#腐蚀膨胀，先做腐蚀再做膨胀
erode=cv2.erode(mask,None,iterations=2)
dilate=cv2.dilate(erode,None,iterations=2)

h, s, v = cv2.split(hsv)
h_red = copy.deepcopy(h)
h_green = copy.deepcopy(h)
for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 255:
            h_red[i, j] += 60

for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 255:
            h_green[i, j] += 120
            
red_background =  cv2.cvtColor(cv2.merge((h_red, s, v)), cv2.COLOR_HSV2BGR)
green_background =  cv2.cvtColor(cv2.merge((h_green, s, v)), cv2.COLOR_HSV2BGR)

cv2.imshow('red_background', red_background)
cv2.imshow('green_background', green_background)
cv2.waitKey(0)
cv2.destroyAllWindows() 
