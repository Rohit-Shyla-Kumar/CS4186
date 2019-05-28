# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:02:49 2019

@author: rohit

FOR CS4186 Assignment 1
"""
#importing the necessary packages
from PIL import Image
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math
import scipy.ndimage
from scipy import spatial


#This is the matrix we will use to generate the filters
mat = np.zeros(shape=(31,31))
mat[15][15] = 1

#Helper functions
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


#PART 1
        
#Question A
        
        
#4 Gaussian filters 
i=0
gaussFilters=[]
sigmas = [1,math.sqrt(2),2,2*math.sqrt(2)]
for i in range(4):
    gaussFilters.append(scipy.ndimage.gaussian_filter(mat,sigmas[i],0))
    showImg(gaussFilters[i])


#8 LOG (Laplacian of Gaussian) 
sigmas=[math.sqrt(2),2,2*math.sqrt(2),4,3*math.sqrt(2),6,6*math.sqrt(2),12]
laplaces = []
for i in range(8):
    laplaces.append(scipy.ndimage.filters.gaussian_laplace(mat,sigmas[i]))
    showImg(laplaces[i])


#o	18 x-directional first and second derivation of Gaussian filters 
sigmas=[1,math.sqrt(2),2]
angles = [0,30,60,90,120,150]
orders = [1,2]
firstDervFilts = []
for i in orders:
    for angle in angles:
        for sigma in sigmas:
            #Rotate by angle -> filter -> show ->append to list
            rot_mat = scipy.ndimage.rotate(mat,angle,reshape= True)
            derv_gauss1 = scipy.ndimage.gaussian_filter(rot_mat,sigma=(1*sigma,3*sigma),order=(i,0))
            print("Angle: "+str(angle)+" "+"Sigma: "+str(sigma)+" "+str(derv_gauss1[15][15]))
            showImg(derv_gauss1)
            firstDervFilts.append(derv_gauss1)

#Filter bank ready
allFilters = gaussFilters+laplaces+firstDervFilts


#Question B


#start with the two query images
panda_orig = cv2.imread("query_imgs/00.jpg")
collo_orig = cv2.imread("query_imgs/01.jpg")
#Grayscale
panda_gray = cv2.cvtColor(panda_orig,cv2.COLOR_BGR2GRAY)
collo_gray = cv2.cvtColor(collo_orig,cv2.COLOR_BGR2GRAY)
orig_img = [panda_gray,collo_gray]
#Convolve with 48 filters and store in lists
panda_results = []
collo_results = []
i=0
for flt in allFilters:
    temp = applyConv(flt,panda_gray)
    temp = cv2.convertScaleAbs(temp)
    showImg(temp)
    show(temp)
    if(i<4):
        s = "filter"+str(i)+"_panda.jpg"
        i+=1
        cv2.imwrite(s,temp)
    panda_results.append(temp)

i = 0
for flt in allFilters:
    temp = applyConv(flt,collo_gray)
    temp = cv2.convertScaleAbs(temp)
    showImg(temp)
    show(temp)  
    if i <4:
        s = "filter"+str(i)+"_collo.jpg"
        i+=1
        cv2.imwrite(s,temp)
    collo_results.append(temp)
        
   
#Compute mean and variance and store in lists , also find max - panda.jpg
panda_means = []
panda_var = []
panda_mean_max = 0
panda_var_max = 0
panda_mean_max_ind = 0
panda_var_max_ind = 0
for i in range(48):
    m = np.mean(panda_results[i])
    v = np.var(panda_results[i])
    if m>panda_mean_max:
        panda_mean_max = m
        panda_mean_max_ind = i
    if v>panda_var_max:
        panda_var_max = v
        panda_var_max_ind = i
    panda_means.append(m)
    panda_var.append(v)

#Compute mean and varianceand store in lists , also find max - colloseum.jpg
collo_means = []
collo_var = []
collo_mean_max = 0
collo_var_max = 0
collo_mean_max_ind = 0
collo_var_max_ind = 0
for i in range(48):
    m = np.mean(collo_results[i])
    v = np.var(collo_results[i])
    if m>collo_mean_max:
        collo_mean_max = m
        collo_mean_max_ind = i
    if v>collo_var_max:
        collo_var_max = v
        collo_var_max_ind = i
    collo_means.append(m)
    collo_var.append(v)


#prepare 96 dimension vector
q1_data = panda_means+panda_var
q2_data = collo_means+collo_var




#PART 2 IMAGE RANKING


#Question A

#Load images -> grayscale -> filter -> compute mean and var -> apply distance formula
data_imgs=[]
rgb_data = []
#load and grayscale
for i in range(10):
    s = "data/0"+str(i)+".jpg"
    rgb_data.append(cv2.cvtColor(cv2.imread(s),cv2.COLOR_BGR2RGB))
    data_imgs.append(cv2.cvtColor(cv2.imread(s),cv2.COLOR_BGR2GRAY))
#    show(data_imgs[i])

for i in range(10,20):
    s = "data/"+str(i)+".jpg"
    rgb_data.append(cv2.cvtColor(cv2.imread(s),cv2.COLOR_BGR2RGB))
    data_imgs.append(cv2.cvtColor(cv2.imread(s),cv2.COLOR_BGR2GRAY))
#    show(data_imgs[i])
    
#filtered
data_means = []
data_var = []
for img in data_imgs:
    temp_mean = []
    temp_var = []
    for flt in allFilters:
        temp = applyConv(flt,img)
        temp = cv2.convertScaleAbs(temp)
        m = np.mean(temp)
        v = np.var(temp)
        temp_mean.append(m)
        temp_var.append(v)
    data_means.append(temp_mean)
    data_var.append(temp_var)
    
data_results=[]
for i in range(20):
    data_results.append(data_means[i]+data_var[i])
    
    
#Find eucledian dist for pandas
distances_panda = []
for lis in data_results:
    distance = 0
    for i in range(96):
        distance+=math.pow((q1_data[i]-lis[i]),2)
    distances_panda.append(math.sqrt(distance))

#Find eucledian dist for pandas
distances_collo = []
for lis in data_results:
    distance = 0
    for i in range(96):
        distance+=math.pow((q2_data[i]-lis[i]),2)
    distances_collo.append(math.sqrt(distance))





consine_panda = []
consine_collo = []

#Find cosine dist for pandas and collo
for lis in data_results:
    result = 1 - spatial.distance.cosine(q1_data, lis)  
    consine_panda.append(result)
    result = 1 - spatial.distance.cosine(q2_data, lis)  
    consine_collo.append(result)
    
    
    
# Question B
    
#convert query images to RGB
rgb_queries = [cv2.cvtColor(panda_orig,cv2.COLOR_BGR2RGB),cv2.cvtColor(collo_orig,cv2.COLOR_BGR2RGB)]

#calc hists for queries
hist_queries = []
for img in rgb_queries:
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    hist_queries.append(hist)
    
#calc hists for data
hist_data = []
for img in rgb_data:
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    hist_data.append(hist)

#eucledian dists b/w hists for pandas 
hist_dist_pandas = []
for f in hist_data:
    distance = 0
    for i in range(0,512):
        distance+=math.pow(hist_queries[0][i]-f[i],2)
    hist_dist_pandas.append(distance)
    
#eucledian dists b/w hists for collo
hist_dist_collo = []
for f in hist_data:
    distance = 0
    for i in range(0,512):
        distance+=math.pow(hist_queries[1][i]-f[i],2)
    hist_dist_collo.append(distance)
    
    
#cosine distance for queries

hist_consine_panda = []
hist_consine_collo = []

#Find cosine dist for pandas and collo
for lis in hist_data:
    result = 1 - spatial.distance.cosine(hist_queries[0], lis)  
    hist_consine_panda.append(result)
    result = 1 - spatial.distance.cosine(hist_queries[1], lis)  
    hist_consine_collo.append(result)
