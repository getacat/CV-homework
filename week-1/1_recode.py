# encoding=utf-8
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 读取灰度图像
img_gray = cv2.imread('alita.jpg', 0)
cv2.imshow('alita_gray', img_gray)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img_gray)

# 读取图像
img = cv2.imread('alita.jpg')
cv2.imshow('alita', img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img)

# 图像的类型
print(img.dtype)
# 图像的形状
print(img.shape)
#剪切图像
img_crop = img[:100,:100]
print(img_crop.shape)
# BGR
b,g,r = cv2.split(img)
cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# 修改图像颜色
def random_light_color(img):
    b, g, r = cv2.split(img)
    
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        b[b > lim] = 255
        b[b < lim] = (b_rand + b[b < lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        b[b > lim] = 255
        b[b < lim] = (b_rand + b[b < lim]).astype(img.dtype)
        
    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        g[g > lim] = 255
        g[g < lim] = (g_rand + g[g < lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - b_rand
        g[g > lim] = 255
        g[g < lim] = (g_rand + g[g < lim]).astype(img.dtype)
        
    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        r[r > lim] = 255
        r[r < lim] = (r_rand + r[r < lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        r[r > lim] = 255
        r[r < lim] = (r_rand + r[r < lim]).astype(img.dtype)
        
    img_merge = cv2.merge((b, g, r))
    return img_merge  

img_random_color = random_light_color(img)
cv2.imshow('random',img_random_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# gamma correction
img_dark = cv2.imread('dark.jpg')

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i/255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)
    
img_brighter = adjust_gamma(img_dark, 2)
cv2.imshow('img_dark', img_dark)
cv2.imshow('img_brighter', img_brighter)
key = cv2.waitKey() 
if key == 27:
    cv2.destroyAllWindows()

# histogram
img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[1]*0.5), int(img_brighter.shape[0]*0.5)))
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# convert the yuv image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
plt.hist(img_brighter.flatten(), 256, [0,256], color = 'r')
plt.hist(img_output.flatten(), 256, [0,256], color = 'g')

cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_output)
key = cv2.waitKey() 
if key == 27:
    cv2.destroyAllWindows()


# Similarity Transform
M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 30, 0.5)
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# Affine Transform
cols = img.shape[1]
rows = img.shape[0]
pts1 = np.float32([[0, 0], [cols - 1, 0],[0, rows - 1]])
pts2 = np.float32([[cols*0.2, rows*0.1], [cols*0.9, rows*0.2], [cols*0.1, rows*0.9]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
# Perspective Transform
img_road = cv2.imread('road.jpg')
pts1 = np.float32([[150, 328], [300, 328], [190, 257], [280, 257]])
pts2 = np.float32([[150, 328], [300, 328], [150, 200], [300, 200]])
M = cv2.getPerspectiveTransform(pts1, pts2)
img_warp = cv2.warpPerspective(img_road, M, (img_road.shape[1], img_road.shape[0]))

cv2.startWindowThread()
cv2.imshow('similarity', img_rotate)
cv2.imshow('affine', dst)
cv2.imshow('perspective', img_warp)
cv2.imshow('img', img)
key = cv2.waitKey() 
if key == 27:
    cv2.destroyAllWindows()
