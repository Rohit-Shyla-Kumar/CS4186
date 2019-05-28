# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:28:26 2019

@author: rohit
"""

#part a of this script is partially adapted from https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html

#importing libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt


##############################################################################
#PART A
##############################################################################

#reading the image and creating a mask of the same size with all zeros
horse = cv2.imread('data/grabcut/a.jpg')
pizza = cv2.imread('data/grabcut/b.jpg')
pig = cv2.imread('data/grabcut/c.jpg')
imgs = [horse,pizza,pig]

#The images have been resized as the grabcut algorithm is too slow for large images and
#the results are not very different

#Original rects
#horseRect = (1340,600,2940,2300)
#pizzaRect = (1075,800,3675,2775)
#pigRect = (540,710,3810,2510)

#Resized Rects
horseRect = (430,190,455,530)
pizzaRect = (320,275,1100,825)
pigRect = (155,220,560,1000)
rects = [horseRect,pizzaRect,pigRect]


#Create masks
horseMask = np.zeros(horse.shape[:2],np.uint8)
pizzaMask = np.zeros(pizza.shape[:2],np.uint8)
pigMask = np.zeros(pig.shape[:2],np.uint8)
masks = [horseMask,pizzaMask,pigMask]

#empty arrays to store fgd and bgd models 
bgdModel = [np.zeros((1,65),np.float64)]*3
fgdModel = [np.zeros((1,65),np.float64)]*3

#number of iterations the grabcut algorithm will run for
iterations = 5

    
for i in range(3):
    cv2.grabCut(imgs[i],masks[i],rects[i],bgdModel[i],fgdModel[i],iterations,cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(imgs[i],masks[i],rects[i],bgdModel[i],fgdModel[i],iterations,cv2.GC_INIT_WITH_MASK)

#list to store the grabcut results
cutImgs = [] 
for i in range(3):
    #convert all 2's to 0's (possible background pixels to sure background pixels)
    mask2 = np.where((masks[i]==2)|(masks[i]==0),0,1).astype('uint8')
    #multiply with the original image to cancel out the background as 0 x anything = 0
    cutImgs.append(imgs[i]*mask2[:,:,np.newaxis])
    #display the image on the console
    plt.imshow(cutImgs[i]),plt.colorbar(),plt.show()


#write out 
cv2.imwrite('a-grabcut.jpg',cutImgs[0]) 
cv2.imwrite('b-grabcut.jpg',cutImgs[1])
cv2.imwrite('c-grabcut.jpg',cutImgs[2])







##THIS SECTION OF THE CODE HAS BEEN SPLIT INTO MULTIPLE SCRIPTS TO MAKE IT MORE READABLE

##############################################################################
#PART B
##############################################################################




##############################################################################
#Disparity Map
##############################################################################


#see scripts coneDisparityMap.py, teaDisparityMap.py and bookDisparityMap.py


##############################################################################
#Relative Height Estimation
##############################################################################


#see scripts coneHeightEstimate.py, teaHeightEstimate.py and bookHeightEstimate.py


