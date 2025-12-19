# Directional Object Detection in Aerial Images

In this repo, we provide the code for DOD method. The DOD code was mainly based on [Ultralytics YOLOV8.1.24](https://github.com/ultralytics/ultralytics). We will present a short description about how to train, evaluate, and run inference with DOD. We also present a few addtional scripts used for data processing, e.g., creating the dataset and image rectification. The pretrained models are also available. Feel free to open an issue, if something does not work.

## Abstract

Object detection in remote sensing consists in recognizing and localizing objects of interest on the Earth’s surface. Classical object detectors must deal with a wide range of distinct side-views, generally oriented upward due to gravity, while objects in aerial images appear in a quasi-standard bird-eye view with an arbitrary orientation and low resolution on the ground plane. This naturally leads to the question: Can we detect the direction of an object along with its location and object category? So-called oriented bounding boxes (OBB) became standard in remote sensing due to the crowded scenes, especially for rotated and elongated objects (e.g., parking cars in a parking lot). OBBs localizes objects by a minimal enclosing box, thus oriented refers to a rotation angle of a standard BB for a tight enclosing of the detected object with respect to the image axes. However, this bounding box orientation does not correspond to the orientation of the detected object, e.g., the front of an airplane or a car, the stem of a leaf, which provides valuable information for aerial image analysis. Herein, we propose a novel method, the Directed Object Detector (DOD), which is capable of detecting the object’s direction together with its minimal enclosing OBB. This is integrated into an object detector to create a single end-to-end neural network. Experimental validation confirms the state of the art performance on both close and far-range remote sensing images in man-made and natural environments. Furthermore, we prove the advantage of DOD over the OBB approach in an image rectification application, which is not solvable with OBB.

<p align="center">
  <img src="assets/arch_horz_simplified3_withimage.jpg"> <br>
</p>

## Datasets

We provide 3 datasets, formatted to be useable out-of-the-box with the DOD model.

Download with:

[OHD-SJTU dataset](http://rocon.utcluj.ro/~levente/download/public/dod/ohdsjtu_dod.zip)

<p align="center">
  <img src="assets/P480_0450_2700.jpg" width="400"> <br>
</p>

[OHD-SJTU occluded dataset](http://rocon.utcluj.ro/~levente/download/public/dod/ohdsjtu_occluded_dod.zip) for validation only

<p align="center">
  <img src="assets/P480_0450_2700_occluded.jpg" width="400"> <br>
</p>

[VEDAI dataset](http://rocon.utcluj.ro/~levente/download/public/dod/vedai_dod.zip) for validation only

<p align="center">
  <img src="assets/00000027_co.png" width="400"> <br>
</p>

[vine leaves dataset](http://rocon.utcluj.ro/~levente/download/public/dod/leaves_dod.zip) the contours, and a few sampled leaves from a laboratory environment are also included.

<p align="center">
  <img src="assets/Cluj_Aug03_DJI_0562_f_00150.jpg" width="400"> <br>
</p>

You have to have your dataset prepared for training and evaluation. The dataset structure should look like this:

```
|-- /path/to/dataset/
    |-- images
        |-- train
        |-- val
    |-- labels
        |-- train
        |-- val
```

For each image, there should be a `.txt` file in the respectiv `label` folder (same name as the image), and for each annotation file, every object is in a new line, in the following format: `class x1 y1 x2 y2 x3 y3 x4 y4 cosDirAngle sinDirAngle`. The `class` is an integer representing the number of the class, the coordinates `x1,y1,...` are normalized coordinate values of the 4 corner points of the oriented bounding box (ellipses has to be converted into this format), while the `cosDirAngle` and `sinDirAngle` are the cosine and sine values of the direction angle (this angle follows the axes of the pixel values, which means, that as the angle grows, the object will be rotated in a clockwise direction on the image, because on images the `Y` axis points downwards).

Do not forget to have the required `yaml` file for your dataset (e.g., 1ultralytics/cfg/datasets/custom/OHD-SJTU-L.yaml1).

## Environment setup

We recommend to use a conda environment for this code. Most necessary packages are in the requirements.txt file. Make sure NOT to install ultralytics from pip, as this repository contains direct modifications into that code, and installing it would create a confusing environment.

## Pre trained models

DOD model trained on the OHD-SJTU-L dataset: [model](http://rocon.utcluj.ro/~levente/download/public/dod/ohdsjtu_best.pt)

DOD model trained on the Leaf dataset: [model](http://rocon.utcluj.ro/~levente/download/public/dod/leaves_best.pt)

## Evaluation

For evaluating a model, use the `script_val.py`.

If you have inferenced labels, you can find out more using `data_processing_scripts/compare_dod.py` script (Note: for this to properly work, you need to have an annotation file from your predictions for each image, however, `script_predict_dod.py` does not create one if there is no detections for a given image. For this run the `data_processing_scripts/check_label_image.py` script.)

If you have the inference result from the HDDet method, you can convert those types of data into the required type using the `data_processing_scripts/hddet_xywhr2bb.py` script.

## Image rectification

Once you have the necessary dataset for image rectificaiton, you can create rectified images using `rectify_leaves.py` script.

### Dataset preprocessing

The original datasets can be found at: [OHD-SHTU](https://thinklab.sjtu.edu.cn/OHD-SJTU.html) and [VEDAI](https://downloads.greyc.fr/vedai/).

For OHD-SJTU you can use the given script in their Developement tools, to spit the images into smaller images. The VEDAI dataset is slightly differently organized, for that use the `data_processing_scripts/vedai2dobb.py` script.

There are also a few other usefule scripts:

If you want to check that your annotations are correct, you can use the `data_processing_scripts/plot_direction.py` script.

You might want to enlarge the variations on your dataset with flipping your images using thet `data_processing_scripts/flip_images.py` script (this also takes care for the labels).

For the OHD-SJTU datasets, you can introduce occlusions onto the image using the `data_processing_scripts/dpt_occlusion.py` script.

To annotate the direction of the leaves, you need MATLAB, with the `data_processing_scripts/matlab/label_leaf_orientation_many.m` script. For other Leaves dataset related question, please open an issue.

### Training

For training the DOD method, you should run `the script_train_dod.py` script. There you can choose the dataset (path to the config file), the representation, and the direction loss type.

### Acknowledgments

<sup>
This work was partially supported by the grants TKP2021-NVA-09 and K135728 of the National Research, Development and Innovation Fund (NKFIH) of Hungary; the HAS Domus scholarship; the ATLAS project funded by the EU CHIST-ERA program (CHIST-ERA-23-MultiGIS-02) and NKFIH under grant 2024-1.2.2-ERA-NET-2025-00020; the project Romanian Hub for Artificial Intelligence-HRIA, Smart Growth, Digitization and Financial Instruments Program, MySMIS no. 351416. 
</sup>