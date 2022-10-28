# Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images

This is the official implementation of Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images. <br />

**Conference Paper** <br />
> [Cross-scale Attention Guided Multi-instance Learning for Crohn's Disease Diagnosis with Pathological Images](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_3) <br />
> Deng, Ruining and Cui, Can and Remedios, Lucas W and Bao, Shunxing and Womick, R Michael and Chiron, Sophie and Li, Jia and Roland, Joseph T and Lau, Ken S and Liu, Qi and others <br />
> *MMMI 2022* <br />

## Abstract
Multi-instance learning (MIL) is widely used in the computer-aided interpretation of pathological Whole Slide Images (WSIs) to solve the lack of pixel-wise or patch-wise annotations. Often, this approach directly applies "natural image driven" MIL algorithms which overlook the multi-scale (i.e. pyramidal) nature of WSIs. Off-the-shelf MIL algorithms are typically deployed on a single-scale of WSIs (e.g., 20x magnification), while human pathologists usually aggregate the global and local patterns in a multi-scale manner (e.g., by zooming in and out between different magnifications). In this study, we propose a novel cross-scale attention mechanism to explicitly aggregate inter-scale interactions into a single MIL network for Crohn's Disease (CD), which is a form of inflammatory bowel disease. The contribution of this paper is two-fold: (1) a cross-scale attention mechanism is proposed to aggregate features from different resolutions with multi-scale interaction; and (2) differential multi-scale attention visualizations are generated to localize explainable lesion patterns. By training ~250,000 H&E-stained Ascending Colon (AC) patches from 20 CD patient and 30 healthy control samples at different scales, our approach achieved a superior Area under the Curve (AUC) score of 0.8924 compared with baseline models.<br /> 


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
