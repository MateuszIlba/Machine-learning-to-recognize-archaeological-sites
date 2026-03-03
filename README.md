# Machine-learning-to-recognize-archaeological-sites
The repository contains a trained model for selecting archaeological sites based on prepared data - aerial photos combined with information from DTM and SAR radar images.

The Mask R-CNN architecture is an evolutionary extension of the Faster R-CNN model, introducing a parallel branch for predicting a binary mask for each object instance. This model's data processing is characterized by high spatial precision, which is crucial for instance segmentation tasks. During the training phase, the model optimizes a complex loss function L (loss). The TensorFlow environment enables an efficient implementation of this process through distributed graph computation. Thanks to modules such as tf.data and the TensorFlow Object Detection API, the training process is optimized for hardware accelerators (GPU/TPU), allowing for processing large datasets (e.g., MS COCO) while maintaining gradient stability.

This research was funded by the National Science Centre, Poland under Grant no. 2024/08/X/ST10/00587
