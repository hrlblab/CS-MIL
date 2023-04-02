# Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images

### [[Project Page]](https://github.com/hrlblab/CS-MIL)   [[Journal paper]](https://arxiv.org/pdf/2206.13632.pdf) [[MMMI 2022 paper]](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_3) <br />


This is the official implementation of Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images. <br />

**Journal Paper** <br />
> [Cross-scale Multi-instance Learning for Pathological Image Diagnosis](https://ieeexplore.ieee.org/document/10079171) <br />
> Ruining Deng, Can Cui, Lucas W. Remedios, Shunxing Bao, R. Michael Womick, Sophie Chiron, Jia Li, Joseph T. Roland, Ken S. Lau, Qi Liu, Keith T. Wilson, Yaohong Wang, Lori A. Coburn, Bennett A. Landman, and Yuankai Huo <br />
> *Under review* <br />


**Conference Paper** <br />
> [Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_3) <br />
> Ruining Deng, Can Cui, Lucas W. Remedios, Shunxing Bao, R. Michael Womick, Sophie Chiron, Jia Li, Joseph T. Roland, Ken S. Lau, Qi Liu, Keith T. Wilson, Yaohong Wang, Lori A. Coburn, Bennett A. Landman, and Yuankai Huo <br />
> *MMMI 2022* <br />

![Overview](https://github.com/hrlblab/CS-MIL/blob/main/Cross-scale.png)<br />
![Pipeline](https://github.com/hrlblab/CS-MIL/blob/main/Relativework.png)<br />
![Cross-scale attention map on a CD WSI](https://github.com/hrlblab/CS-MIL/blob/main/AttentionMap.png)<br />
![Toy dataset](https://github.com/hrlblab/CS-MIL/blob/main/Toydataset.png)<br />
![Cross-scale attention map on toy dataset](https://github.com/hrlblab/CS-MIL/blob/main/ToydatasetResults.png)<br />


## Abstract
Analyzing high resolution whole slide images (WSIs) with regard to information across multiple scales poses a significant challenge in digital pathology. Multi-instance learning (MIL) is a common solution for working with high resolution images by classifying bags of objects (i.e. sets of smaller image patches). However, such processing is typically performed at a single scale (e.g., 20X magnification) of WSIs, disregarding the vital inter-scale information that is key to diagnoses by human pathologists. In this study, we propose a novel cross-scale MIL algorithm to explicitly aggregate inter-scale relationships into a single MIL network for pathological image diagnosis. The contribution of this paper is three-fold: (1) A novel cross-scale MIL (CS-MIL) algorithm that integrates the multi-scale information and the inter-scale relationships is proposed; (2) A toy dataset with scale-specific morphological features is created and released to examine and visualize differential cross-scale attention; (3) Superior performance on both in-house and public datasets is demonstrated by our simple cross-scale MIL strategy.<br /> 


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
