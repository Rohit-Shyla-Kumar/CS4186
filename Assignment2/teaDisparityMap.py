# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:56:49 2019

@author: rohit
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

#helper function to draw epilines
#here, we draw lines on img1 using the keypoints of img2
def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


#import cones images in grayscale
tea_l = cv2.imread("data/relative_height/2_a.jpg", cv2.IMREAD_GRAYSCALE)
tea_r = cv2.imread("data/relative_height/2_b.jpg", cv2.IMREAD_GRAYSCALE)

#Use SIFT to detect features and compute the descriptors 
sift = cv2.xfeatures2d.SIFT_create()


#find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(tea_l,None)
kp2, des2 = sift.detectAndCompute(tea_r,None)

#draw keypoints for both images and write out the images
img = cv2.drawKeypoints(tea_l,kp1,outImage=np.array([]), color=(0, 0, 255))
plt.imshow(img),plt.colorbar(),plt.show()
cv2.imwrite("tea_l_keypoints.jpg",img)
img =cv2.drawKeypoints(tea_r,kp2,outImage=np.array([]), color=(0, 0, 255))
plt.imshow(img),plt.colorbar(),plt.show()
cv2.imwrite("tea_r_keypoints.jpg",img)

#FLANN parameters
#Here we use FLANN instead of a brute force matcher for better results
#FLANN matches the two sets of key points
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 100)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

#ratio test as per Lowe's paper
ratio = 0.7
for i,(m,n) in enumerate(matches):
    if m.distance < ratio*n.distance:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# cv2.drawMatchesKnn expects list of lists as matches
# We only give it the first 500 matches for convienence 
img3 = cv2.drawMatchesKnn(tea_l,kp1,tea_r,kp2,good[:500],None,flags=2)

cv2.imwrite('tea_matches.jpg',img3)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

#Get the fundamental matrix
F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)


# Obtainment of the rectification matrix and use of the warpPerspective to transform them...
pts1 = pts1[:,:][mask.ravel()==1]
pts2 = pts2[:,:][mask.ravel()==1]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)


#Here we draw the epilines of the right images on the left and the left on the right


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(tea_l,tea_r,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(tea_r,tea_l,lines2,pts2,pts1)


#show the epilines 
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

#Write the files to memory 
cv2.imwrite('tea_l_epilines.jpg',img5)
cv2.imwrite('tea_r_epilines.jpg',img3)



p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))

retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,tea_l.shape[:2])

dst_l = cv2.warpPerspective(tea_l,rectmat1,tea_l.shape[:2])
dst_r = cv2.warpPerspective(tea_r,rectmat2,tea_r.shape[:2])

#calculation of the disparity
stereo = cv2.StereoBM_create()
disp = stereo.compute(dst_r.astype(np.uint8), dst_l.astype(np.uint8)).astype(np.float32)
plt.imshow(disp)
plt.colorbar()
plt.clim(0,100)
plt.savefig("tea_disp.png")