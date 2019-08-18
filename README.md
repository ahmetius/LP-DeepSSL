# LabelProp-SSDL
This repository contains the code for the following paper:

> A. Iscen, G. Tolias, Y. Avrithis, O. Chum. "Label Propagation for Deep Semi-supervised Learning", CVPR 2019

This code generally follows [Mean Teacher Pytorch implementation](https://github.com/CuriousAI/mean-teacher/tree/master/pytorch). Python 3 is required.

##  Data pre-processing:

### CIFAR-10
As in the original Mean Teacher repository, run the following command:

```
>> ./data-local/bin/prepare_cifar10.sh
```

### CIFAR-10
Run the similar command for CIFAR-100, taken from the [fastswa-semi-sup](https://github.com/benathi/fastswa-semi-sup/tree/master/data-local/bin) repository:
```
>> ./data-local/bin/prepare_cifar100.sh
```

### Mini-Imagenet
We took the Mini-Imagenet dataset hosted in [this repository](https://github.com/gidariss/FewShotWithoutForgetting) and pre-processed it. You can download the dataset in .pkl format [here](). Please extract the contents in the following directory:
```
>> ./data-local/images/
```

##  Running the experiments:

There are two stages to our method. 

Stage 1 consists of training a network with known labels only. Following command can be run to reproduce this experiment:
```
>> python train_stage1.py --exclude-unlabeled=True --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=False --isL2=True --dataset=$DATASET
```
where ```$NOLABELS``` is the number of labeled points, ```$GPUID``` is the GPU to be used (0 by default), ```$SPLITID``` is the ID of the split to be used (10-19 in our experiments), and ```$DATASET``` is the name of the dataset (cifar10, cifar100, or miniimagenet).

After the training for Stage 1 is completed, run the following command for Stage 2, which resumes the training from the model trained in Stage 1, but this time with pseudo-labels using the entire dataset:

```
>> python train_stage2.py --labeled-batch-size=50 --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=False --isL2=True --dataset=$DATASET
```

##  Combining with Mean Teacher:
Use the following commands for combining Mean Teacher with our method:

Stage 1:
```
>> python train_stage1.py --labeled-batch-size=50 --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=True --isL2=False --dataset=$DATASET
```

Stage 2:
```
>> python train_stage2.py --labeled-batch-size=50 --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=True --isL2=False --dataset=$DATASET
```








