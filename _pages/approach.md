---
layout: page
title:  "Approach"
---

The existing implementations for visual relationship detection output all possible object-object relationships when fed an image. We intend to detect specific object pairs and a relationship predicate between them that match the given textual input while at the same time try and minimize the number of erroneous pairs our model outputs. The model to achieve the same is described in the subsequent sections. 

	1.1. The first step would be to identify a <subject, predicate, object> triplet from the given text input. We will be using the Stanford NLP toolkit/spacy for this task.
	
	1.2. We would then find all possible object pairs in the image. Current state-of-the-art models for object detection achieve impressive performance by leveraging deep convolutional neural networks. We plan to use the Fast RCNN pipeline for this module. 
	
	1.3 After finding all possible object pairs, we find object pairs that correspond to the <s,o> detected in the triplet obtained from step 1.1. 
	
	1.4 We also introduce a thresholded intra-bounding box distance filter to discard pairs of <s,o> that are spatially distant in the image. This is to keep in line with our limited set of spatial predicates.

In this project, we propose a few-shot learning based approach to learn the predicate relationship in images. We do this by using a pre-trained convolutional neural network (VGG16) and optimizing a triplet-loss function. Triplet loss is a loss function for neural networks where a baseline (anchor) input is compared to a positive input and a negative input. The distance from the baseline input to the positive input is minimized, and the distance from the baseline input to the negative input is maximized. The input to this model would be an image of our <s,o> pair. The output of this model would be class probabilities for our limited set of predicates. Due to computational limitations and time constraints, we will be using a limited set of spatial predicates to test our approach.

## Tasks:

	Object Detection: Returning bounding boxes around entities in the image that look like the ‘subject’ and the ‘object’ from the input description.
	
	Object Pair Filtering: Filtering out object pairs in the image that do not have a semantically meaningful relationship between them (spatially distant objects).
	
	Neural Network Model: Based on the positions of relevant bounding boxes with respect to each other, detect the probability of the relationship corresponding to the input predicate.