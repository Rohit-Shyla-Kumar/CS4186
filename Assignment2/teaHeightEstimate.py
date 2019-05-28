# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:28:42 2019

@author: rohit
"""

import cv2
import numpy as np 
import math
from matplotlib import pyplot as plt


#reading the image and creating a mask of the same size with all zeros
tea_l = cv2.imread('data/relative_height/2_a.jpg')
tea_r = cv2.imread('data/relative_height/2_b.jpg')


tea_lRect = (301,363,1523,1063)
tea_rRect = (338,424,1504,1086)



#Create masks
tea_lMask = np.zeros(tea_l.shape[:2],np.uint8)
tea_rMask = np.zeros(tea_r.shape[:2],np.uint8)

masks = [tea_lMask,tea_rMask]

#empty arrays to store fgd and bgd models 
bgdModel = [np.zeros((1,65),np.float64)]*3
fgdModel = [np.zeros((1,65),np.float64)]*3


iterations = 5


cv2.grabCut(tea_l,tea_lMask,tea_lRect,bgdModel[0],fgdModel[0],iterations,cv2.GC_INIT_WITH_RECT)
cv2.grabCut(tea_l,tea_lMask,tea_lRect,bgdModel[0],fgdModel[0],iterations,cv2.GC_INIT_WITH_MASK)

cv2.grabCut(tea_r,tea_rMask,tea_rRect,bgdModel[1],fgdModel[1],iterations,cv2.GC_INIT_WITH_RECT)
cv2.grabCut(tea_r,tea_rMask,tea_rRect,bgdModel[1],fgdModel[1],iterations,cv2.GC_INIT_WITH_MASK)
    
#convert all 2's to 0's (possible background pixels to sure background pixels)
mask2 = np.where((tea_lMask==2)|(tea_lMask==0),0,1).astype('uint8')
mask3 = np.where((tea_rMask==2)|(tea_rMask==0),0,1).astype('uint8')
#multiply with the original image to cancel out the background as 0 x anything = 0
cutImgs = []
cutImgs.append(tea_l*mask2[:,:,np.newaxis])
cutImgs.append(tea_r*mask3[:,:,np.newaxis])
#display the images
plt.imshow(cutImgs[0]),plt.colorbar(),plt.show()
plt.imshow(cutImgs[1]),plt.colorbar(),plt.show()
    

cv2.imwrite('tea_r-grabcut.jpg',cutImgs[0])
cv2.imwrite('tea_l-grabcut.jpg',cutImgs[1])


#import grabcut cones images in grayscale
tea_l = cv2.imread("tea_l-grabcut.jpg", cv2.IMREAD_GRAYSCALE)
tea_r = cv2.imread("tea_r-grabcut.jpg", cv2.IMREAD_GRAYSCALE)

#Use SIFT to detect features and compute the descriptors 
sift = cv2.xfeatures2d.SIFT_create()


#find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(tea_l,None)
kp2, des2 = sift.detectAndCompute(tea_r,None)

#draw keypoints for both images and write out the images
img = cv2.drawKeypoints(tea_l,kp1,outImage=np.array([]), color=(0, 0, 255))
plt.imshow(img),plt.colorbar(),plt.show()
cv2.imwrite("tea_l_grabcut_keypoints.jpg",img)
img =cv2.drawKeypoints(tea_r,kp2,outImage=np.array([]), color=(0, 0, 255))
plt.imshow(img),plt.colorbar(),plt.show()
cv2.imwrite("tea_r__grabcut_keypoints.jpg",img)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 100)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        
# cv2.drawMatchesKnn expects list of lists as matches
# We only give it the first 500 matches for convienence 
img3 = cv2.drawMatchesKnn(tea_l,kp1,tea_r,kp2,good,None,flags=2)

cv2.imwrite('tea_grabcut_matches.jpg',img3)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

ratio = 0.0
ctr = 0
for i in range(len(pts1)):
        ratio += pts1[i][1]/pts2[i][1]
        ctr+=1
print("Ratio of left to right = "+str(ratio/ctr))