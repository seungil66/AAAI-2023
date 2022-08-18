# AAAI-2023

# Multi-step-training-framework
This is the PyTorch implementation of the paper "Multi-step training framework using Sparsity training for Efficient utilization of Accumulated new data".

## Enviornment details
Ubuntu 18.04.5    
CUDA 10.2   
Python version 3.7    
Pytorch version 1.9.0   
torchvision 0.10.0    

## Split dataset
For Single-step experiment, we split CIFAR-100 and Tiny-ImageNet datasets by 1:1 and used as previous data and new data.   
For Multi-step experiment, we split CIFAR-100 and Tiny-ImageNet dataset by 1:1:1:1 and used as Data1, Data2, Data3, and Data4.  

## Multi-step training using CIFAR-100

To conduct multi-step training with CIFAR-100 dataset on ResNet110, run this command Sequentially.   

### Step 1
**(1) : Training using Data1**
```
python train.py --arch resnet110 --batch_size 256 --cifar 100 
```
**(2) : SST using Data1**
```
Python sp_train.py --arch resnet110 --batch_size 256 --resume path_to_result_of_(1) --s regularization_strength --ratio target_r
```
### Step 2 (Single-step)
**(3) : AOD using Data2**
```
python aod.py --arch resnet110 --batch_size 256 --cifar 100 --resume path_to_result_of_(2) --ratio target_r_of_(2)
```
**(4) : SST with MSL using Data2**
```
python multi_sp_train.py --arch resnet110 --batch_size 256 --cifar 100 --resume path_to_result_of_(3) --s regularization strength --ratio target_r
```
### Step 3 (Multi-step)
**(5) : AOD using Data3**
```
python aod.py --arch resnet110 --batch_size 256 --cifar 100 --resume path_to_result_of_(4) --ratio target_r_of_(4)
```
**(6) : SST with MSL using Data3**
```
python multi_sp_train.py --arch resnet110 --batch_size 256 --cifar 100 --resume path_to_result_of_(5) --s regularization strength --ratio target_r
```
### Step 4 (Multi-step)
**(7) : AOD using Data4**
```
python aod.py --arch resnet110 --batch_size 256 --cifar 100 --resume path_to_result_of_(6) --ratio target_r_of_(6)
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
