import cv2
import numpy as np

input_image = cv2.imread('mouse_pad.jpg')
input_image = cv2.resize(input_image, (400,550),interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=2000)


keypoints, descriptors = orb.detectAndCompute(gray_image, None)

final_keypoints = cv2.drawKeypoints(gray_image, keypoints,input_image,(0,255,0))

cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey()
# Press esc to end