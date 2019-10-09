---
layout: page
title:  "Approach"
---

Input to the system will be a textual description in the form of a triplet ‘(subject, predicate, object)’ for instance, ‘laptop on desk’. The objective is to produce bounding boxes around the objects involved in the relationship. The model to achieve the same is described in the subsequent sections. 

Training phase, 

	1.1. The first step is to find all object pairs in training images using object detection algorithms like RCNN, fast RCNN or YOLO. So, an image with a cowboy riding a horse gives object pairs like ‘cowboy and horse’, ‘cowboy and hat’, ‘horse and saddle’ etc. 

	1.2. Visual module - After finding all possible object pairs from the training data, the next step is to find the possibility of the detected object pairs belonging to the corresponding ground truth relationship label, which is of the form <i,j,k> where i and k are object classes and j is the predicate. Only one type of detected object pairs (that have the relationship as specified by the ground truth) will have the highest probability of belonging to the label. This is achieved by training CNN classifiers. 

	1.3 Language module - This module finds word embeddings of the detected objects and the matching relationship and projects them to common vector space as a relationship vector. After this, similar relationship vectors are found between different object-predicate-object triplets. This is done to match never-seen-before relationships like ‘man riding an elephant’ to the ones already encountered like ‘man riding horse’. 

Testing phase

	Given a test image and a textual description of a visual relationship between two objects, the task is to detect these objects in the image and use the visual and language modules to learn appropriate features that match the relationship specified through the textual description. Spatial filtering is also performed between the detected object pairs to eliminate unlikely pairs that don’t have meaningful relationship between them. --example??. As output, the object pairs that best match the relationship specified are highlighted with bounding boxes in the image.   
		
		
Tasks:

	Object detection: make bounding boxes around all entities in the image that look like the ‘subject’ and the ‘object’ given in the input description.
	
	Object pair filtering: A neural network to filter out object pairs in the image that do not have a visibly meaningful relationship between them (spatially distant objects).
	
	Model: based on the positions of these bounding boxes with respect to each other, detect if the relationship corresponding to the input predicate exists between the entities in question.