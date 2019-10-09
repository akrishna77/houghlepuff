---
layout: page
title:  "Project Proposal"
---

### Problem Statement

The objective of this project is to detect objects of interest in a given image and highlight the relationships between these objects. Almost all real-world objects have visual relationships between them. The input to the system is an image along with a textual description of an object-object relationship and the output is the image with bounding boxes around the objects that correspond to the textual input. The output would also include a scene graph representing all object-object relationships in the image. For instance, given an input “person on a skateboard”, the system would give as output: 


### Approach

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

### Experiments & Evaluation

For implementation, we use Python 3.x version and keras or PyTorch framework with required libraries. 
Dataset used- We use the Visual Relationship Dataset which has 5000 images with 100 object categories and 70 predicates. In total, it has 37,993 relationships and 6,672 relationship types. The relationships have been classified into many categories, some of which are action, preposition, comparative, verb etc.  Some examples of relationships included in the dataset are  ‘person kick ball’, ‘person on top of ramp’, ‘motorcycle with wheel’, ‘man riding horse’ etc.  

We may use bits of existing code from the following projects- 

https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection (for detecting object pairs)

https://github.com/matterport/Mask_RCNN (for detecting object pairs)

https://github.com/yikang-li/vg_cleansing (for creating the language module)

https://github.com/jz462/Large-Scale-VRD.pytorch (for visual relationship understanding)

https://github.com/doubledaibo/drnet_cvpr2017 (for visual relationship detection[2])

https://github.com/GriffinLiang/vrd-dsr (for visual relationship detection[3])

The existing implementations for visual relationship detection output all possible object-object relationships when fed a test image. We intend to detect specific object pairs and a relationship predicate between them that match the given textual input while at the same time try and minimize the number of erroneous pairs our model outputs. 

A specific case we want to handle is as follows:
Given an input ‘man holding pen’ and an image where there is a man and a pen lying on the notebook next to him, our system should not display bounding boxes around the man or the pen because there is no ‘holding’ relationship. As another example, for the image below, the input ‘man on horse’ should correctly ignore the man walking in front of the horse. This relationship should be discarded during the filtering stage.  

We use recall (the percentage of total relevant results correctly classified by our system) (the ability of our model to find all the relevant cases within the dataset) as a performance metric. This is to account for the possibility that some object-object relationships may not have been seen during the training phase.  

We will test our model on two datasets: VRD and SVG.

If our model correctly displays bounding boxes with an acceptable precision, we’ll consider it a success. We expect to exploit the correlation between visual and language cues of an image to find accurate visual relationships. We also wish to experiment with how spatial context of objects and the relationships they are involved in differ. For example, a ‘man riding a horse’ and ‘man standing next to a horse’ have the same objects - ‘man’ and ‘horse’ with similar context and spatial features, yet the relationships between them are different.  


### References

1. Cewu Lu, Ranjay Krishna, Michael Bernstein and Li Fei-Fei “Visual Relationship Detection with Language Priors”
2. Bo Dai, Yuqi Zhang and Dahua Lin “Detecting Visual Relationships with Deep Relational Networks” 
3. Kongming Liang, Yuhong Guo, Hong Chang and Xilin Chen “Visual Relationship Detection with Deep Structural Ranking” In The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18) 
4. Stephan Baier, Yunpu Ma, and Volker Tresp “Improving Visual Relationship Detection usingSemantic Modeling of Scene Descriptions”


