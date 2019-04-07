# coding=utf-8
import cv2
import numpy as np

# 不会做，写了一个随机剪切旋转的
img = cv2.imread('alita.jpg')

def random_rotation(img, num):
    cols = img.shape[1]
    rows = img.shape[0]
    out = []
    for i in range(num):
        angle = np.random.randint(-30, 30)
        scale = np.random.uniform(0.5, 2)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        out.append(cv2.warpAffine(img, M, (cols, rows)))
    return out

def random_crop(imgs):
    cols = img.shape[1]
    rows = img.shape[0]
    out = []
    for i in imgs:
        x_1 = np.random.randint(0, cols/8)
        x_2 = np.random.randint(7*cols/8, cols)
        y_1 = np.random.randint(0, rows/8)
        y_2 = np.random.randint(7*rows/8, rows)
        out.append(i[x_1:x_2, y_1:y_2])
    return out

def adjust_gamma(imgs, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    out = []
    for i in range(256):
        table.append(((i/255)**invGamma)*255)
        table = np.array(table).astype("uint8")
    for i in imgs:
        out.append(cv2.LUT(img, table))
    return out

def equalize(imgs):
    out = []
    for i in imgs:
        for j in range(3):
            i[:, :, j] = cv2.equalizeHist(i[:, :,j])
        out.append(i)
    return out

n = 3
imgs = random_rotation(img, n)
imgs_2 = random_crop(imgs)
imgs_3 = equalize(imgs_2)
for i in imgs_3:
    n = n - 1
    print(i.shape[0],i.shape[1])
    cv2.imshow(str(n), i)
    key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()