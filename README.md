# ADMultiView

## Introduction
we proposed a novel classification method based on the fusion of multiview 2D and 3D convolutions for MRI-based AD diagnosis.
Specifically, we first use multiple sub-networks to extract the local slice-level feature of each slice in different dimensions.
Then a 3D convolution network was used to extract the global subject-level information of MRI. Finally, local and global
information were fused to acquire more discriminative features. Experiments conducted on the ADNI-1 and ADNI-2 dataset
demonstrated the superiority of this proposed model over other state-of-the-art methods for their ability to discriminate AD and
Normal Controls (NC). Our model achieves 90.2% and 85.2% of accuracy on ADNI-2 and ADNI-1 respectively, thus it can be
effective in AD diagnosis.
## dataset
The dataset used in this study was obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) that is avaiable at http://adni.loni.usc.edu/ .

## Pre-process
All MRIs were prepocessed by a standard pipeline in CAT12 toolbox which is avaiable at http://dbm,neuro.uni-jena.de/cat/.

## Prerequisites
Linux python 3.7 Pytorch version 1.2.0 NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 10.0.61

## Note
Please cite our paper if you use this code in your own work.

