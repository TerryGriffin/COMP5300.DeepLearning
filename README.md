# COMP5300.DeepLearning

This repository contains the code associated with the final project for COMP.5300 Deep Learning,
Spring 2020.

The code depends on the Pytorch version 1.3.1 (https://pytorch.og) and Detectron2 (https://github.com/facebookresearch/detectron2)

The file layout is:

root
    
* doc - the final report and presentation
* grayscale_transfer - the code for the grayscale transfer learning experiment
* xray_detection -  the code for the TB manifestation detection experiments
    * config - the configuration files for training models for Airspace Consolidation, 
    Cavitation, Lymphadenopathy, and Pleural Effusion, for R-CNN, Mask R-CNN, Cascade R-CNN, 
    Cascade Mask R-CNN with ResNet-50 and ResNet-101 backbone networks.
     
    