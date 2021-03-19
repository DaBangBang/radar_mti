import numpy as np
import cv2

# vidcap = cv2.VideoCapture('D:\data_signal_MTI\project_util_3\prediction_result\prediction_vdo_robot_3_fix.mp4')
# success,image = vidcap.read()
# count = 0
# while success:
#   if count % 5 == 0:
#     cv2.imwrite("C:/Users/nakorn-vision/Desktop/pic_robot/frame_%d.jpg" % count, image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1

img5 = cv2.imread('C:/Users/nakorn-vision/Desktop/pic_robot/frame_2340.jpg')
img4 = cv2.imread('C:/Users/nakorn-vision/Desktop/pic_robot/frame_2335.jpg')
img3 = cv2.imread('C:/Users/nakorn-vision/Desktop/pic_robot/frame_2330.jpg')
img2 = cv2.imread('C:/Users/nakorn-vision/Desktop/pic_robot/frame_2325.jpg')
img1 = cv2.imread('C:/Users/nakorn-vision/Desktop/pic_robot/frame_2320.jpg')
img = cv2.imread('C:/Users/nakorn-vision/Desktop/pic_robot/frame_2315.jpg')
h, w, _ = img1.shape
print(h, w)
img = img[200:720,300:850]
img1 = img1[200:720,300:850]
img2 = img2[200:720,300:850]
img3 = img3[200:720,300:850]
img4 = img4[200:720,300:850]
img5 = img5[200:720,300:850]

cv2.imwrite("C:/Users/nakorn-vision/Desktop/pic_robot/frame_a.jpg", img)
cv2.imwrite("C:/Users/nakorn-vision/Desktop/pic_robot/frame_b.jpg", img1)
cv2.imwrite("C:/Users/nakorn-vision/Desktop/pic_robot/frame_c.jpg", img2)
cv2.imwrite("C:/Users/nakorn-vision/Desktop/pic_robot/frame_d.jpg", img3)
cv2.imwrite("C:/Users/nakorn-vision/Desktop/pic_robot/frame_e.jpg", img4)
cv2.imwrite("C:/Users/nakorn-vision/Desktop/pic_robot/frame_f.jpg", img5)

# The function requires that the two pictures must be the same size
# img2 = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
#print img1.shape, img2.shape
#alpha, beta, gamma adjustable
# alpha = 0.8
# beta = 1-alpha
# gamma = 0
# img_add = cv2.addWeighted(img1, alpha, img, beta, gamma)
# img_add = cv2.addWeighted(img2, alpha, img_add, beta, gamma)
# img_add = cv2.addWeighted(img3, alpha, img_add, beta, gamma)
# cv2.namedWindow('addImage')
cv2.imshow('img_add',img)
cv2.waitKey()
cv2.destroyAllWindows()