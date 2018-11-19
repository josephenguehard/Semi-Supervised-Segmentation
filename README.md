# Semi-Supervised Deep Embedded Clustering applied to the iSeg 2017 challenge

This repository contains the code based on our extension of Deep Embedding Clustering to semi-supervised training. This method is applied to the iSeg 2017 dataset, which can be download here: http://iseg2017.web.unc.edu/.

### Folders

The main code is in the Jupyter Notebook file. The file functions.py stores many utils functions to load, preprocess and save the data, and the file clustering_layer.py contains a Keras layer which is added on top of our network. The training and testing sets have to be added to the main directory in the subfolders datasets/iSeg2017/iSeg-2017-Training and datasets/iSeg2017/iSeg-2017-Testing

### Libraries
The code requires the following configuration

- jupyter == 1.0.0
- keras == 2.1.6
- nibabel == 2.3.1
- python == 2.7.12
- sckit-learn == 0.20.0
- tensorflow == 1.3.0

