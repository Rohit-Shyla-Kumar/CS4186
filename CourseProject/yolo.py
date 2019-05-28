# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:26:51 2019

@author: rohit

description - using YOLOv2 for object detection on the VOC2012 dataset
"""
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


classesFile = "models/voc.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 


###Create a model using the opencv dnn module and analyze each image of our dataset
    
modelConfiguration = "models/yolo/yolov2.cfg";
modelWeights = "models/yolo/yolov2-voc.weights";
#load model
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)

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
    image = cv2.imread(path+'\\'+"{:05d}.jpg".format(i))
    (H, W) = image.shape[:2]
     
    #only the output layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
     
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    
    #Forward pass in the convolutional network
    layerOutputs = net.forward(ln)
    
     
    
    #Create lists of detected bounding boxes, confidences, class IDs, and results
    boxes = []
    confidences = []
    classIDs = []
    result = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence (i.e., probability) of
    		# the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
            
    		# filter out weak predictions by ensuring the detected confidence is greater than the threshold
    		if confidence > confThresh:
    			# yolo returns the center of the bounding box, width and height
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))
     
    			# update our list of bounding box coordinates, confidences and class IDs
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)
                
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.7,0.3)
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        
    	# loop over the indexes we are keeping
        for i in idxs.flatten():
            
            #add it to our list of results
            result.append(classes[classIDs[i]])
            
            
            
            #For visualization, add boxes and lables
#            (x, y) = (boxes[i][0], boxes[i][1])
#            (w, h) = (boxes[i][2], boxes[i][3])
#            color = (255,0,0)
#            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#            text = "{} ".format(classes[classIDs[i]])
#            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            
    #append all strong preditions for the image       
    results.append(result)
    
    #For Visualization and debugging
    # show the output image
#    cv2.imshow("Image", image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
    
# how long did it take for 17125 images
end = time.time()
print("YOLOv2 took {:.6f} seconds".format(end - start))




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
print("YOLOv2 Results:\n")

print("recall = "+str(recall))
print("precision = "+str(precision))
print("f1 = "+str(f1))
print("ratio of objects detected = "+str(true_ratio))