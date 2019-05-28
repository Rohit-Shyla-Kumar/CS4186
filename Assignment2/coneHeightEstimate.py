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
cones_l = cv2.imread('data/relative_height/1_a.jpg')
cones_r = cv2.imread('data/relative_height/1_b.jpg')


cones_lRect = (94,240,400,320)
cones_rRect = (70,250,400,320)



#Create masks
cones_lMask = np.zeros(cones_l.shape[:2],np.uint8)
cones_rMask = np.zeros(cones_r.shape[:2],np.uint8)

masks = [cones_lMask,cones_rMask]

#empty arrays to store fgd and bgd models 
bgdModel = [np.zeros((1,65),np.float64)]*3
fgdModel = [np.zeros((1,65),np.float64)]*3


iterations = 5


cv2.grabCut(cones_l,cones_lMask,cones_lRect,bgdModel[0],fgdModel[0],iterations,cv2.GC_INIT_WITH_RECT)
cv2.grabCut(cones_l,cones_lMask,cones_lRect,bgdModel[0],fgdModel[0],iterations,cv2.GC_INIT_WITH_MASK)

cv2.grabCut(cones_r,cones_rMask,cones_rRect,bgdModel[1],fgdModel[1],iterations,cv2.GC_INIT_WITH_RECT)
cv2.grabCut(cones_r,cones_rMask,cones_rRect,bgdModel[1],fgdModel[1],iterations,cv2.GC_INIT_WITH_MASK)
    
#convert all 2's to 0's (possible background pixels to sure background pixels)
mask2 = np.where((cones_lMask==2)|(cones_lMask==0),0,1).astype('uint8')
mask3 = np.where((cones_rMask==2)|(cones_rMask==0),0,1).astype('uint8')
#multiply with the original image to cancel out the background as 0 x anything = 0
cutImgs = []
cutImgs.append(cones_l*mask2[:,:,np.newaxis])
cutImgs.append(cones_r*mask3[:,:,np.newaxis])
#display the images
plt.imshow(cutImgs[0]),plt.colorbar(),plt.show()
plt.imshow(cutImgs[1]),plt.colorbar(),plt.show()
    

cv2.imwrite('cones_r-grabcut.jpg',cutImgs[0])
cv2.imwrite('cones_l-grabcut.jpg',cutImgs[1])


#import grabcut cones images in grayscale
cones_l = cv2.imread("cones_l-grabcut.jpg", cv2.IMREAD_GRAYSCALE)
cones_r = cv2.imread("cones_r-grabcut.jpg", cv2.IMREAD_GRAYSCALE)

#Use SIFT to detect features and compute the descriptors 
sift = cv2.xfeatures2d.SIFT_create()


#find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(cones_l,None)
kp2, des2 = sift.detectAndCompute(cones_r,None)

#draw keypoints for both images and write out the images
img = cv2.drawKeypoints(cones_l,kp1,outImage=np.array([]), color=(0, 0, 255))
plt.imshow(img),plt.colorbar(),plt.show()
cv2.imwrite("cones_l_grabcut_keypoints.jpg",img)
img =cv2.drawKeypoints(cones_r,kp2,outImage=np.array([]), color=(0, 0, 255))
plt.imshow(img),plt.colorbar(),plt.show()
cv2.imwrite("cones_r__grabcut_keypoints.jpg",img)

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
img3 = cv2.drawMatchesKnn(cones_l,kp1,cones_r,kp2,good,None,flags=2)

cv2.imwrite('cones_matches.jpg',img3)

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