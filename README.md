# AAAI-2023

# Multi-step-training-framework
This is the PyTorch-SSD implementation of the paper "Uncertainty-based One-phase Learning to Enhance Pseudo-Label Reliability for Semi-supervised Object Detection".

## Enviornment details
Ubuntu 18.04.5    
CUDA 9.2   
Python version 3.7    
Pytorch version 1.2.0   
torchvision 0.4.0    

## Dataset
Using VOC2007 as labeled dataset and VOC2012 as unlabeled dataset.  

### Training step
**Training**
```
CUDA_VISIBLE_DEVICES=[] python train_ssd_gsm_ucfilter.py 
```
**Training with adaptive filtering**
```
CUDA_VISIBLE_DEVICES=[] python train_ssd_gsm_ucfilter.py --adaptive_filtering=True
```
### Evaluation step
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
