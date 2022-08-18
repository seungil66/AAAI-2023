# AAAI-2023

This is the PyTorch-SSD implementation of the paper "Uncertainty-based One-phase Learning to Enhance Pseudo-Label Reliability for Semi-supervised Object Detection".

## Enviornment details
Ubuntu 18.04.5    
CUDA 9.2   
Python version 3.7    
Pytorch version 1.2.0   
torchvision 0.4.0    

## Dataset
Using VOC2007 as labeled dataset and VOC2012 as unlabeled dataset.  

## Training step
**Training**
```
CUDA_VISIBLE_DEVICES=[] python train_ssd_gsm_ucfilter.py  

# You can search for pseudo-label update point in train_ssd_gsm_ucfilter.py with keyword [update]
CUDA_VISIBLE_DEVICES=[] python train_ssd_gsm_ucfilter.py --adaptive_filtering=True
```

**Training with proposed method**
  - FN solution (uc weighted loss)
     - Search key word : scale
     - Weighted by uncertainty for all negative sample
  - FP solution (adaptive filtering)
     - Search key word : adaptive
     - If you want to compare static filtering, find the static keyword and uncommend it.
  - Pseudo label update 
     - Search key word : update
     - 




## Evaluation step
**Eval mAP(%)**
```
python eval_voc_gsm.py --trained_model=weights/ssd_300_120000.pth
```


## Result
We report the average Top-1 accuracy of three runs.

### Single-step training results
|    Dataset    |   Network   | Previous data Top-1(%) | New data Top-1(%) |
|:-------------:|:-----------:|:----------------------:|:-----------------:|
|   CIFAR-100   |  ResNet110  |          63.19         |       69.71       |
| Tiny-ImageNet | MobileNetv2 |          54.69         |       59.00       |

### Multi-step training results
|    Dataset    |   Network   | Data1 Top-1(%) | Data2 Top-1(%) | Data3 Top-1(%) | Data4 Top-1(%) |
|:-------------:|:-----------:|:--------------:|:--------------:|:--------------:|:--------------:|
|   CIFAR-100   |  ResNet110  |      53.45     |      59.80     |      63.18     |      65.27     |
| Tiny-ImageNet | MobileNetv2 |      45.02     |      50.40     |      53.09     |      54.64     |
