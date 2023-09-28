# Cross-scale Multi-instance Learning for Pathological Image Diagnosis

### [[Pipeline Docker]](https://hub.docker.com/repository/docker/ddrrnn123/cs-mil/)  [[Project Page]](https://github.com/hrlblab/CS-MIL)   [[Journal paper]](https://arxiv.org/pdf/2304.00216.pdf) [[MMMI 2022 paper]](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_3) <br />


This is the official implementation of Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images. <br />

**Journal Paper** <br />
> [Cross-scale Multi-instance Learning for Pathological Image Diagnosis](https://arxiv.org/pdf/2304.00216.pdf) <br />
> Ruining Deng, Can Cui, Lucas W. Remedios, Shunxing Bao, R. Michael Womick, Sophie Chiron, Jia Li, Joseph T. Roland, Ken S. Lau, Qi Liu, Keith T. Wilson, Yaohong Wang, Lori A. Coburn, Bennett A. Landman, and Yuankai Huo <br />
> *Under review* <br />


**Conference Paper** <br />
> [Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_3) <br />
> Ruining Deng, Can Cui, Lucas W. Remedios, Shunxing Bao, R. Michael Womick, Sophie Chiron, Jia Li, Joseph T. Roland, Ken S. Lau, Qi Liu, Keith T. Wilson, Yaohong Wang, Lori A. Coburn, Bennett A. Landman, and Yuankai Huo <br />
> *MMMI 2022* <br />

```diff
+ We release an inference pipeline as a single Docker.
```

## Quick Start

#### WSI examples

The examples of WSI can be downloaded [here](https://drive.google.com/drive/folders/14PfBtjDNXZCLxcWu0hfTMQwIUeGmARSc?usp=drive_link).  <br /> 

#### Get Our Docker Image and Run The Docker with GPU

```
sudo docker pull ddrrnn123/cs-mil:2.0
docker run --rm -v [/Data2/CS-MIL_data]/input:/input/:ro -v [/Data2/CS-MIL_data]/output:/output --gpus all -it ddrrnn123/cs-mil:2.0
```

You may put your WSIs in the "input" folder and change the dirname inside of "[]" to your local root. <br />

You can also refer the source code of the whole pipeline in [run_inference.py](https://github.com/hrlblab/CS-MIL/blob/main/CS-MIL_Docker/src/run_inference.py) for the step-by-step process, which are <br /> 
- Step1. Get tiles (with foreground segmentation); <br />
- Step2. Embedding the patches by SimSiam pre-trained models at different scales; <br />
- Step3. Clustering the features; <br />
- Step4. Get CD classification by pretrained CS-MIL models. <br />

## Abstract
![Overview](https://github.com/hrlblab/CS-MIL/blob/main/Cross-scale.png)<br />
![Pipeline](https://github.com/hrlblab/CS-MIL/blob/main/Relativework.png)<br />
![AttentionMap](https://github.com/hrlblab/CS-MIL/blob/main/AttentionMap.png)<br />

Analyzing high resolution whole slide images (WSIs) with regard to information across multiple scales poses a significant challenge in digital pathology. Multi-instance learning (MIL) is a common solution for working with high resolution images by classifying bags of objects (i.e. sets of smaller image patches). However, such processing is typically performed at a single scale (e.g., 20X magnification) of WSIs, disregarding the vital inter-scale information that is key to diagnoses by human pathologists. In this study, we propose a novel cross-scale MIL algorithm to explicitly aggregate inter-scale relationships into a single MIL network for pathological image diagnosis. The contribution of this paper is three-fold: <br /> 

(1) A novel cross-scale MIL (CS-MIL) algorithm that integrates the multi-scale information and the inter-scale relationships is proposed; <br /> 
(2) A toy dataset with scale-specific morphological features is created and released to examine and visualize differential cross-scale attention; <br /> 
(3) Superior performance on both in-house and public datasets is demonstrated by our simple cross-scale MIL strategy.<br /> 

## Deployment on CD dataset
#### Self-supervised Learning for Patch Embedding
(1) Run [main_mixprecision.py](https://github.com/hrlblab/CS-MIL/blob/main/Emb_Clustering_Code/main_mixprecision.py) to train SimSiam models at different scales. <br /> 
(2) Run [get_features_simsiam_256.py](https://github.com/hrlblab/CS-MIL/blob/main/Emb_Clustering_Code/get_features_simsiam_256.py) (20x) (same for 512.py (10x), 1024.py (5x)) to extract features from patches. <br /> 

#### K-mean Clustering
Run [create_kmeans_features_local_singleresolution.py](https://github.com/hrlblab/CS-MIL/blob/main/Emb_Clustering_Code/create_kmeans_features_local_singleresolution.py) to get k-mean clustering results from features. <br /> 

#### Training and Testing
(1) Run [MIL_global_Stage1_Training.py](https://github.com/hrlblab/CS-MIL/blob/main/Train_Test_Code/MIL_global_Stage1_Training.py) to train the model. <br /> 
(2) Run [MIL_global_Stage1_Testing.py](https://github.com/hrlblab/CS-MIL/blob/main/Train_Test_Code/MIL_global_Stage1_Training.py) to test the model. <br /> 

## Toydataset
To assess the effectiveness of the cross-scale attention mechanism, we evaluated CS-MIL using two toy datasets that represent distinct morphological patterns at different scales in digital pathology. These datasets were selected to simulate different scenarios and test the functionality of our approach.<br />

The figure below shows the patches for training in the two datasets (Micro-anomaly dataset and Macro-anomaly dataset). <br />

(1) The  micro white crosses pattern only appear on positive patches at 20x maganification in the micro-anomaly dataset.  <br />
(2) The macro anomaly (ellipse) is easily recognized at 5x with larger visual fields in macro-anomaly dataset.  <br />
All of the patches are extracted from normal tissue samples in Unitopatho dataset. Two datasets were released to measure the generalization of the cross-scale designs for digital pathology community. <br /> 

The patches and testing regions are avaliable at [here](https://drive.google.com/drive/folders/1PvWi4lmA0bPeLZFRxDqYFftth69srIyn?usp=sharing). <br />

<img src='https://github.com/hrlblab/CS-MIL/blob/main/Toydataset.png' align="center" height="530px"> 

The proposed method accurately differentiates distinctive patterns at different scales in a stable manner. Figure below displays the cross-scale attention maps at the instance level and multiple scales.  <br />

(1) For the Micro-anomaly dataset, the instance attention successfully highlights positive regions with higher attention scores in corresponding regions at 20x.  <br />
(2) For the Macro-anomaly dataset, the instance attention correctly locates ellipses instead of circles with higher attention scores at 5x.  <br />
(3) The box plots on the right panel show the attention score distribution at different scales, proving that the cross-scale attention mechanism provides reliable scores at different scales.<br />

![Cross-scale attention map on toy dataset](https://github.com/hrlblab/CS-MIL/blob/main/ToydatasetResults.png)<br />


## Deployment on Toydataset
#### Data Preprocessing
Run [MIL_bag_generation.py](https://github.com/hrlblab/CS-MIL/blob/main/Toydataset_Code/data_processing/MIL_bag_generation.py) to generate the bags for trainingset and validationset. <br /> 

#### Training and Testing
(1) Run [MIL_main_DeepSurv_dataset1.py](https://github.com/hrlblab/CS-MIL/blob/main/Toydataset_Code/cs-mil-toydataset/MIL_main_DeepSurv_dataset1.py) (same for dataset2.py) to train the model. <br /> 
(2) Run [MIL_main_DeepSurv_batch_dataset1_getattention.py](https://github.com/hrlblab/CS-MIL/blob/main/Toydataset_Code/cs-mil-toydataset/MIL_main_DeepSurv_batch_dataset1_getattention.py) (same for dataset2.py) to test the model and get the attention scores. <br /> 


## Citation
```
@inproceedings{deng2022cross,
  title={Cross-Scale Attention Guided Multi-instance Learning for Crohn’s Disease Diagnosis with Pathological Images},
  author={Deng, Ruining and Cui, Can and Remedios, Lucas W and Bao, Shunxing and Womick, R Michael and Chiron, Sophie and Li, Jia and Roland, Joseph T and Lau, Ken S and Liu, Qi and others},
  booktitle={International Workshop on Multiscale Multimodal Medical Imaging},
  pages={24--33},
  year={2022},
  organization={Springer}
}
```
