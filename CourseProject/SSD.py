# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:07:41 2019

@author: rohit

description - using ssd for object detection on the VOC2012 dataset
"""

#importing libraries 
import numpy as np
import cv2
import time
import xml.etree.ElementTree as ET


###First, we extract the ground truth results for our dataset

#path to annotations
path = "VOCdevkit\\VOC2012\\Annotations"
anns = []


# Read every file in directory
for i in range(1,17126):
    #open each xml file
    tree = ET.parse(path+'\\'+'{:05d}.xml'.format(i))
    root = tree.getroot()
    result = []
    for child in root:
        #look for object tags in the xml
        if(child.tag == "object"):
            result.append(child[0].text)    #append the name of the tag 
    anns.append(result) #append all object names for that image



###Read class names 


classesFile = "models/voc-ssd.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    
    


###Create a model using the opencv dnn module and analyze each image of our dataset
    

#load model
net = cv2.dnn.readNetFromCaffe('models/ssd/ssdPrototxt.txt','models/ssd/ssdModel.caffemodel')

#minimum threshold of confidence required to classify detection
confThresh = 0.6

#path to images directory with 17125 images
path = "VOCdevkit\\VOC2012\\JPEGImages"

#to track how long the whoel process will take
start = time.time()
results = []

# Read every file in directory
for i in range(1,17126):
    # load the input images and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(path+'\\'+"{:05d}.jpg".format(i))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,(300, 300), 127.5)
    
    
    # pass the blob through the network and obtain the detections and predictions    
    net.setInput(blob)
    detections = net.forward()
    
    
    result = []
    
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
    	# extract the confidence associated with the prediction
    	confidence = detections[0, 0, i, 2]
    
    	# filter out weak detections by ensuring the `confidence` is greater than the threshold
    	if confidence > confThresh:
            
            #append the strong prediction to the result list for this image
            idx = int(detections[0, 0, i, 1])
            result.append(classes[idx])
            
            
            #to visualize the images for debugging
#            label = "{} %.2".format(classes[idx],confidence)
#            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#            (startX, startY, endX, endY) = box.astype("int")
#            cv2.rectangle(image, (startX, startY), (endX, endY),(255,0,0), 2)
#            y = startY - 15 if startY - 15 > 15 else startY + 15
#            cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


    #append all strong preditions for the image
    results.append(result)
    
    
    #Visualization
    #show the output image
#    cv2.imshow("Output", image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    

# how long did it take for 17125 images
end = time.time()
print("SSD took {:.6f} seconds".format(end - start))




#Calculating results and scores 


true_positives = 0
false_positives = 0
totalObjs = 0

#Iterate over all the results 
for i in range(1,17125):
    #for each object detected
    for obj in results[i]:
        #if the object is present in the ground truth it is a true positive
        if(obj in anns[i]):
            true_positives+=1       
        #else, it is a false positive
        else:
            false_positives+=1
    #total number of objects present in the ground truth
    for obj in anns[i]:
        totalObjs+=1
        
#if the object was a part of the ground truth and not in true positives, it is a false negative
false_negatives = totalObjs-true_positives

#Calculating different scoring metrics to rate the performance of the algorithm 
recall = true_positives/(true_positives+false_negatives)
precision = true_positives/(true_positives+false_positives)
f1 = 2*(recall*precision)/(recall+precision)
true_ratio = true_positives/totalObjs


#Printing results
print("SSD Results:\n")

print("recall = "+str(recall))
print("precision = "+str(precision))
print("f1 = "+str(f1))
print("ratio of objects detected = "+str(true_ratio))