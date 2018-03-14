#/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 以灰度模式读入图片，这样img.shape就只有二维。否则还会多一维表示彩色
img=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
img2 = img.copy()
# 指定第二个参数0会导致mathTemplate执行报错：template = cv2.imread('Lena_eyes.png',0)
template = cv2.imread('Lena_eyes.png',cv2.IMREAD_GRAYSCALE)

w, h = template.shape[::-1]

# Apply template Matching
res = cv2.matchTemplate(img2,template,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img2,top_left,bottom_right,255,2)

# 绘制1行2列的grid，图片显示在第二个位置
plt.subplot(122), plt.imshow(img2,cmap = 'gray')
# 使用plt.xticks([])关闭坐标轴刻度
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle("TM_CCOEFF_NORMED")
plt.show()
