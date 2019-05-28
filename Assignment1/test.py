# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:53:15 2019

@author: rohit
"""

#importing the necessary packages
from PIL import Image
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math
import scipy.ndimage

mat = np.zeros(shape=(31,31))
mat[15][15] = 1
def applyConv(fil,img):
    return cv2.filter2D(img,-1,fil)
    
def showImg(image):
    plt.subplot(121),plt.imshow(image),plt.title('Result')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def show(image):
    cv2.imshow("Image",image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        

panda_orig = cv2.imread("query_imgs/00.jpg")
collo_orig = cv2.imread("query_imgs/01.jpg")
panda_gray = cv2.cvtColor(panda_orig,cv2.COLOR_BGR2GRAY)
collo_gray = cv2.cvtColor(collo_orig,cv2.COLOR_BGR2GRAY)
orig_img = [panda_orig,collo_orig]
temp = panda_orig

#preparing filter bank
#gauss1 = scipy.ndimage.gaussian_filter(mat,3,0,mat)
gauss1 = cv2.getGaussianKernel(31,3)
mat = applyConv(gauss1,mat)
showImg(mat)
rot_mat = scipy.ndimage.rotate(mat,30)
showImg(rot_mat)
#gauss2 = cv2.getGaussianKernel(31,math.sqrt(2))
#gauss3 = cv2.getGaussianKernel(31,2)
#gauss4 = cv2.getGaussianKernel(31,2*math.sqrt(2))
#gauss5 = cv2.getGaussianKernel(31,4)
#gauss6 = cv2.getGaussianKernel(31,3*math.sqrt(2))
#gauss7 = cv2.getGaussianKernel(31,6)
#gauss8 = cv2.getGaussianKernel(31,6*math.sqrt(2))
#gauss9 = cv2.getGaussianKernel(31,12)

gauss_filters = [gauss1,gauss2,gauss3,gauss4]
log_filters = [gauss2,gauss3,gauss4,gauss5,gauss6,gauss7,gauss8,gauss9]

for img in orig_img:
    for fil in gauss_filters:
        img = applyConv(fil,img)
        showImg(img)
        
        
for img in orig_img:
    for fil in log_filters:
        img = applyConv(fil,img)
        img = cv2.Laplacian(img,cv2.CV_16S,31)
        img = cv2.convertScaleAbs(img)
#        cv2.imshow("Image",img)
#        cv2.waitKey(0)
        showImg(img)


pandas1 = []
pandas2 = []
for i in range(18):
    pandas1.append(panda_gray)
    pandas2.append(panda_gray)

for i in range(18):
    derv1gauss = scipy.ndimage.filters.gaussian_filter(pandas1[i],sigma = (1,3),order = (1,0),truncate = 14)
    show(derv1gauss)

        
