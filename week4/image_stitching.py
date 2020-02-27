# -*- coding: utf-8 -*-
"""
This module implements the image stitching algorithm and a visualized demo.
It is recommended run the demo in a IDE(e.g. Spyder).
You can also juts type following command to run the visualized demo:
    $ python image_stitching.py
Then it will generate the match image and stitched image in the current folder.
"""

import cv2
import numpy as np

def remove_black_borders(img):
    
    # 先转为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # 再转为二值图
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # findContours()在cv2的不同版本可以可能返回2个或3个输出，下面这行是对所有cv2是普适的
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    
    return img[y:y+h,x:x+w]
    
class ImageStitch():
    
    def __init__(self):
        
        self.sifter = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        self.min_match_cnt = 10
        self.ratio = 0.03
        
    def sift(self, gray_img):
        """
        补充解释一下key_points和descriptors的基本属性
        """
        return self.sifter.detectAndCompute(gray_img, None)
    
    def match(self, descriptors_left, descriptors_right):
        
        goods = []
        matches = self.matcher.knnMatch(descriptors_right, descriptors_left, k=2)
        
        for m1, m2 in matches:
            if m1.distance < self.ratio * m2.distance:
                goods.append(m1)

        return goods
    
    def find_homography(self, key_points_left, key_points_right, goods):
        
        if len(goods) > self.min_match_cnt:
            
            src_points = np.float32([key_points_right[good.queryIdx].pt 
                                     for good in goods]).reshape(-1,1,2)
            dst_points = np.float32([key_points_left[good.trainIdx].pt 
                                     for good in goods]).reshape(-1,1,2)
    
            homo_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            return homo_matrix
        
        else:
            raise RuntimeError("There are no enough match points")
    
    def draw_matches(self, img_right, key_points_right, img_left, key_points_left, goods):
        
        draw_params = dict(matchColor=(0, 0, 255), # 红色匹配线
                       singlePointColor=None,
                       flags=2)
        img_matches = cv2.drawMatches(
            img_right, key_points_right, 
            img_left, key_points_left, goods, None, **draw_params)
        
        cv2.imwrite('img_matches.jpg', img_matches)
        return
    
    def process(self, src_img_left, src_img_right):
        
        # 读入需要拼接的两张图，需要左右按顺序读入
        src_img_left = cv2.imread(src_img_left)
        src_img_right = cv2.imread(src_img_right)
        
        # 转换成灰度图（SIFT算法的需求）
        gray_img_left = cv2.cvtColor(src_img_left, cv2.COLOR_BGR2GRAY)
        gray_img_right = cv2.cvtColor(src_img_right, cv2.COLOR_BGR2GRAY)
        
        # SITF算法，得出关键点及其描述子
        key_points_left, descriptors_left = self.sift(gray_img_left)
        key_points_right, descriptors_right = self.sift(gray_img_right)
        
        # 寻找两张图匹配的描述子
        goods = self.match(descriptors_left, descriptors_right)
        
        self.draw_matches(
                src_img_right, key_points_right,
                src_img_left, key_points_left, goods)
        
        # 得到单应性矩阵
        homo_matrix = self.find_homography(key_points_left, key_points_right, goods)
        
        # 根据单应性矩阵进行仿射变换
        output_size = (src_img_left.shape[1]+src_img_right.shape[1], src_img_left.shape[0])
        img_output = cv2.warpPerspective(src_img_right, homo_matrix, (output_size))
        img_output[0:src_img_left.shape[0], 0:src_img_left.shape[1]] = src_img_left
        
        # 移除黑色区域
        img_output = remove_black_borders(img_output)
        
        cv2.imwrite('img_stitched.jpg', img_output)

def demo():
    stitcher = ImageStitch()
    stitcher.process('airplane_left.jpg', 'airplane_right.jpg')

if __name__ == "__main__":
    demo()