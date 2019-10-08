---
layout: page
title:  "Evaluation"
---

We use recall (the percentage of total relevant results correctly classified by our system) (the ability of our model to find all the relevant cases within the dataset) as a performance metric. This is to account for the possibility that some object-object relationships may not have been seen during the training phase.  

We will test our model on two datasets: VRD and SVG.

If our model correctly displays bounding boxes with an acceptable precision, we’ll consider it a success. We expect to exploit the correlation between visual and language cues of an image to find accurate visual relationships. We also wish to experiment with how spatial context of objects and the relationships they are involved in differ. For example, a ‘man riding a horse’ and ‘man standing next to a horse’ have the same objects - ‘man’ and ‘horse’ with similar context and spatial features, yet the relationships between them are different.  
