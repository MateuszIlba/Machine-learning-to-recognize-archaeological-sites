# Machine-learning-to-recognize-archaeological-sites
The repository contains a trained model for selecting archaeological sites based on prepared data - aerial photos combined with information from DTM and SAR radar images.

The Mask R-CNN architecture is an evolutionary extension of the Faster R-CNN model, introducing a parallel branch for predicting a binary mask for each object instance. This model's data processing is characterized by high spatial precision, which is crucial for instance segmentation tasks. During the training phase, the model optimizes a complex loss function L (loss). The TensorFlow environment enables an efficient implementation of this process through distributed graph computation. Thanks to modules such as tf.data and the TensorFlow Object Detection API, the training process is optimized for hardware accelerators (GPU/TPU), allowing for processing large datasets (e.g., MS COCO) while maintaining gradient stability.

## Model training
The machine learning process was conducted using a specialized training dataset, including archaeological site locations. This dataset consisted of over 1,010 precisely prepared data units, representing a fusion of multi-source remote sensing information.
Each sample consisted of:
- Spectral data (RGB): Standard images in the visible spectrum, providing information on surface texture and color.
- Rao's Q index: A parameter defining the functional diversity and heterogeneity of the landscape, enabling the identification of ecological anomalies that may indicate the presence of anthropogenic structures.
- SAR (Synthetic Aperture Radar) intensity: Microwave data that, thanks to its ability to penetrate vegetation and its sensitivity to terrain microrelief and soil moisture, enables the detection of archaeological features invisible in the optical spectrum.
The use of such a constructed, multimodal dataset allowed the Mask R-CNN model to learn the correlation between subtle changes in biophysical indicators and the physical presence of archaeological remains, which significantly increases the efficiency of automatic site prediction in difficult field conditions.

Monitoring the training process revealed a systematic decrease in the error value, demonstrating effective adaptation of the network weights to the specific characteristics of the multimodal archaeological dataset. After 250 training epochs, the loss function stabilized at a level oscillating around 0.5. This behavior of the error regression carries important technical implications: 
- achieving a level of 0.5 indicates that the model successfully passed the initial generalization phase and precisely adjusted its parameters to detect subtle anomalies in RGB, SAR, and Rao's Q data
- his value, assuming the complexity of the task (fusion of remote sensing and archaeological data), suggests achieving the optimal model operating point. The model managed to minimize classification errors and box and mask localization errors, while maintaining the ability to generalize to data outside the training set
- since the total loss in Mask R-CNN is the sum of the components (L_cls + L_box + L_mask), a score of 0.5 means that each of the network heads achieved high precision, minimizing errors in both site type identification and in accurately outlining its pixel-by-pixel boundaries.

The achieved stabilization of the loss function after 250 epochs confirms that the integrated physical terrain features (SAR intensity) and biological features (Rao’s Q) constitute a coherent input signal that the neural network in the TensorFlow environment can effectively interpret to automatically identify anthropogenic objects.

<img width="1590" height="673" alt="image" src="https://github.com/user-attachments/assets/1ed91a24-c908-4305-a466-5f4b70dd3a4b" />
Fig 1. Decreasing loss value on the training and validation sets (val)
<br/><br/>
The weight matrix of the optimized Mask R-CNN model, developed over 250 training epochs, was exported and secured in an external research data repository. The binary file, containing the complete state of the trained convolutional layers and masking head weights, was deposited on the Zenodo platform https://doi.org/10.5281/zenodo.17053519

<br/><br/><br/>
## This research was funded by the National Science Centre, Poland under Grant no. 2024/08/X/ST10/00587
<img width="3082" height="268" alt="logo-poziom-en" src="https://github.com/user-attachments/assets/3a3cd11d-a41a-4d37-89d1-8c4652f3a946" />

