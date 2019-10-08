---
layout: page
title:  "Experiments"
---

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
