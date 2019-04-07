import cv2

img = cv2.imread('colorful.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,2] = cv2.equalizeHist(img_yuv[:,:,2])
img_yuv[:,:,1] = cv2.equalizeHist(img_yuv[:,:,1])
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output_1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

y, u, v = cv2.split(img_yuv)
u = 255 - u
y = 255 - y
v = 255 - v
merge = cv2.merge((y, u, v))
img_output_2 = cv2.cvtColor(merge, cv2.COLOR_YUV2BGR)

cv2.imshow('out_1', img_output_1)
cv2.imshow('out_2', img_output_2)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()