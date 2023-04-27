# Deep Image Matching Using Local Feature Transformers

The process to reconstruct 3D objects and buildings from images is called Structure-from-Motion (SfM). Typically, these images are captured by skilled operators under controlled conditions, ensuring homogeneous, high-quality data. It is much more difficult to build 3D models from assorted images, given a wide variety of viewpoints, lighting and weather conditions, occlusions from people and vehicles, and even user-applied filters. The first part of the problem is to identify which parts of two images capture the same physical points of a scene, such as the corners of a window. This is typically achieved with local features. Feature matching refers to finding corresponding features from two similar images based on a search distance algorithm. One of the images is considered the source and the other as target, and the feature matching technique is used to either find or derive and transfer attributes from source to target image. The feature matching process generally analyses the source and target’s image topology, detects the feature patterns, matches the patterns, and matches the features within the discovered patterns. The accuracy of feature matching depends on image similarity, complexity, and quality. Normally, a high percentage of successful matching can be achieved using the correct method, while uncertainty and errors may occur and would require post-inspection and corrections. 

The aim of this project is to find matching points between two images (views) of a same scene by obtaining the fundamental matrix, and hence the relative pose between the two images. It is important because it’s the first step for 3D Reconstruction, Simultaneous Localization and Mapping, and Panoramic Stitching.

## Usage

Steps to create environment
```bash
# create a conda environment. Python version 3.9 is necessary.
conda create -n loftr python=3.9

# install relevant libraries
conda install pytorch torchvision kornia einops pandas matplotlib opencv loguru -c pytorch
pip install pytorch_lightning kornia_moons
```
## File Structure
Please make sure you **follow this specific file structure** mentioned here. The dataset and other relevant files can be found [here](https://drive.google.com/drive/folders/1-zAaqigu1OWFG6PhUg51aIcKRk9gkByh?usp=sharing).

```
.
├── depth-masks-imc2022
│  └── depth_maps
├── evaluation-notebook.ipynb
├── image-matching-challenge-2022
│  ├── sample_submission.csv
│  ├── test.csv
│  ├── test_images
│  └── train
├── imc-gt
│  └── train.csv
├── kornia-loftr
│  ├── kornia-0.6.4-py2.py3-none-any.whl
│  ├── kornia_moons-0.1.9-py3-none-any.whl
│  ├── loftr_outdoor.ckpt
│  ├── outdoor_ds.ckpt
│  └── outdoor_ot.ckpt
├── loftrutils
│  ├── einops-0.4.1-py3-none-any.whl
│  ├── LoFTR-master
│  └── outdoor_ds.ckpt
├── README.md
├── train.py
├── training-notebook.ipynb
└── weights
   └── model_weights.ckpt
```
## Dataset

<div align=center>
<img src="https://user-images.githubusercontent.com/39590621/168615651-16a5faaf-d444-4bde-ae53-baf4e97581c2.png">
</div>
<br>

The training set contains thousands of images from 16 locations, all of which are popular tourist attractions. This includes the likes of Buckingham Palace, the Lincoln Memoria, Notre Dame Cathedral, the Taj Mahal, and the Pantheon. In addition to theimages, they provide two csv files. The calibration file contains the camera calibration matrices that are necessary for building fundamental matrices. The pair covisibility file contains the covisibility metric between pairs of images and the ground truth fundamental matrices for each pair. The test set contains 3 image pairs that contestants are to generate fundamental matrices for to demonstrate the submissions. 

## Kornia

<div align=center>
<img src="https://user-images.githubusercontent.com/39590621/168612245-70119dea-53e5-4ea3-b8ba-d27bccfac941.png">
</div>
<br>

Kornia is a differentiable library that allows classical computer vision to be integrated into deep learning models. It consists of a set of routines and differentiable modules to solve generic computer vision problems. At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.Kornea has a golden rule of not having heavy dependencies. Kornia module provides a number of descriptors including the SIFT (Scale-Invariant Feature Transform) descriptor, MKDDescriptor (multiple kerneldescriptor), HardNet descriptor, HardNet8 descriptor, HyNet descriptor, SOSNet (Second-Order Similarity) descriptor, and TFeat descriptor.

## LoFTR

LoFTR (Local Feature TRansformer) is a Framework that provides a method for image feature matching. Instead of performing image processing methods such as image feature detection, description, and matching one by one sequentially, it first establishes a pixel-wise dense match and later refines the matches. In contrast to traditional methods that use a cost volume to search corresponding matches, the framework uses self and cross attention layers from its Transformer model to obtain feature descriptors present on both images. The global receptive field provided by Transformer enables LoFTR to produce dense matches in even low-texture areas, where traditional feature detectors usually struggle to produce repeatable interest points. Furthermore, the framework model comes pre-trained on indoor and outdoor datasets to detect the kind of image being analyzed, with features like self-attention. Hence, it   makes LoFTR outperform other state-of-the-art methods by a large margin. 

<div align=center>
<img src="https://user-images.githubusercontent.com/39590621/168614880-48bb08e3-8553-4d80-b7b1-54175c247d8a.png">
</div>
<br>

LoFTR has the following steps: 

* CNN extracts the coarse-level feature maps Feature A and Feature B, together with the fine-level feature maps created from the image pair A and  B . 
* Then, the created feature maps get flattened into 1-D vectors and are added with the positional encoding that describes the positional orientation of objects present in the input image. The added features are then processed by the Local Feature Transformer (LoFTR) module. 
* Further, a differentiable matching layer is used to match the transformed features, which provide a confidence matrix. The matches are then selected according to the confidence threshold level and mutual nearest-neighbor criteria, yielding a coarse-level match prediction.  
* For every selected coarse prediction made, a local window with size w × w is cropped from the fine-level feature map. Coarse matches are then refined from this local window to a sub-pixel level and considered as the final match prediction.

## Evaluation metric

For this challenge, all the participants are asked to estimate the relative pose of one image with respect to another. Submissions are evaluated on the mean Average Accuracy (mAA) of the estimated poses. Given a fundamental matrix and the hidden ground truth, error in terms of rotation ( , in degrees) and translation (, in meters) is computed. Given one threshold over each, pose as accurate if it meets both thresholds is classified. This is done over ten pairs of thresholds, one pair at a time. 
The percentage of image pairs that meet every pair of thresholds is calculated, and average the results over all thresholds, which rewards more accurate poses. As the dataset contains multiple scenes, which have a different number of pairs, we compute this metric separately for each scene and average it afterwards.

# Results

The mAA value for our model is 0.725 across all the scenes In the dataset.

<div align=center>
<img src="results/res1.png">
<img src="results/res2.png">
<img src="results/res3.png">
</div>

# Observations & Conclusions
* Transformers can provide a much better estimate of the pose between two cameras, compared to traditional methods.
* Because of the positional encoding aspect, Transformers are a very good way to distinguish locally similar features from globally similar features between two images of a scene across wide baseline.
* This enables more robust image matching for multiple computer vision tasks like 3D Reconstruction, SLAM, SfM & Panoramic Stitching. 
