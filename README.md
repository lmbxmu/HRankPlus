# An extension version of our CVPR 2020, oral: HRank: Filter Pruning using High-Rank Feature Map ([Link](https://128.84.21.199/abs/2002.10179)).

### We are now releasing this code under the request of many friends. However, we are not sure if this code is stable. Please contact us if you have found any problem, which would be appreciated. 

### Prior code version can be found [here](https://github.com/lmbxmu/HRank).

### In the next two or three months, we will continue to maintain this repository and the pruned models will be released constantly. 

## Tips

Any problem, please contact the authors via emails: lmbxmu@stu.xmu.edu.cn or ethan.zhangyc@gmail.com, or adding the first author's wechat as friends (id: linmb007 if you are using wechat) for convenient communications. Do not post issues with github as much as possible, just in case that I could not receive the emails from github thus ignore the posted issues.


## Citation
If you find HRank useful in your research, please consider citing:

```
@inproceedings{lin2020hrank,
  title={HRank: Filter Pruning using High-Rank Feature Map},
  author={Lin, Mingbao and Ji, Rongrong and Wang, Yan and Zhang, Yichen and Zhang, Baochang and Tian, Yonghong and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1529--1538},
  year={2020}
}
```

## Running Code

In this code, you can run our models on CIFAR-10 and ImageNet dataset. (Task on object detection is preparing)

For rank generation, the code has been tested by Python 3.6, Pytorch 1.0 and CUDA 9.0 on Ubuntu 16.04. And for evaluating(fine-tuning), we recommend you to run with PyTorch 0.4 for the usage of Thop.


### Rank Generation

```shell
python rank_generation.py \
--dataset [dataset name] \
--data_dir [dataset dir] \
--pretrain_dir [pre-trained model dir] \
--arch [model arch name] \
--limit [batch numbers] \
--gpu [gpu_id]

```
For the ease of reproducibility, we provide the extracted ranks [here](https://drive.google.com/drive/folders/1kwOFEtmUw6jwk_qNpLydwUjlouuexd5R?usp=sharing):


### Model Training

For model training, our code will automatically save the previous-epoch checkpoint file in `checkpoint.pth.tar` and best checkpoint in `model_best.pth.tar` after every epoch. And with the `resume` option, you can resume training from the previous checkpoint under the same job directory.

##### 1. CIFAR-10

```shell
python evaluate_cifar.py \
--data_dir [CIFAR-10 dataset dir] \
--job_dir ./result/[model name]/[folder name] \
--arch [model name](vgg_16_bn, resnet_56, resnet_110, googlenet, densenet_40) \
--rank_conv_prefix [rank folder dir] \
--use_pretrain \
--pretrain_dir [pre-trained model dir] \
--compress_rate [compress rate] \
--gpu [gpu_id]
```

##### 2. ImageNet

```shell
python evaluate.py \
--data_dir [ImageNet dataset dir] \
--use_dali \
--job_dir ./result/[model name]/[folder name] \
--arch [model name](resnet_50, mobilenet_v1, mobilenet_v2) \
--rank_conv_prefix [rank folder dir] \
--lr_type (cos or step) \
--use_pretrain \
--pretrain_dir [pre-trained model dir] \
--compress_rate [compress rate] \
--gpu [gpu_id]
```

##### 3. Optional arguments(ImageNet)
```
optional arguments:
    --data_dir				Dataset directory.
    --job_dir				The directory where the summaries will be stored.
    --rank_conv_prefix 			Prefix directory of rank folder.
    --compress_rate 			Compress rate of each convolutional layer.
    --arch				Architecture of model. default: vgg_16_bn 
    					Optional: resnet_50, mobilenet_v2, mobilenet_v1, vgg_16_bn, resnet_56, resnet_110, densenet_40, googlenet
    --learning_rate			Initial learning rate. default: 0.1
    --lr_type 				Learning rate decay schedule. default: step. optional: step, cos
    --momentum   			Momentum for optimizer. default: 0.9
    --epochs				The number of epochs to train. default: 90
    --batch_size			Batch size for both training and validation. default: 64
    --weight_decay 			The weight decay of loss function. default: 1e-4
    --label_smooth			Label smooth parameter. default: 0.1
    
    --resume				If this parameter exists, model training by resuming from previous ckpt in the same directory.
    --use_dali 				If this parameter exists, use dali module to load ImageNet data.
    --use_pretrain			If this parameter exists, use pretrained model for finetuning.
    --pretrain_dir			Pretrained model directory.
    --test_only				If this parameter exists, only validate the model performance without training.
    --test_model_dir			Test model directory
    --gpu	 			Select gpu to use. default: 0
```


The following are the examples of compression rate setting for several models: 
(Please note that the following compression rates are only used to demonstrate the parameter format, which may not be used in our experiment. For the specific pruning rate, please refer to the configuration files of pruned model in the next section)

|  Model      | Compression Rate |
|:-------------:|:-------------------------:|
| VGG-16-BN | [0.45]\*7+[0.78]\*5 | 
| ResNet-56 | [0.]+[0.18]\*29 | 
| ResNet-110 | [0.]+[0.2]\*2+[0.3]\*18+[0.35]\*36 | 
| GoogLeNet | [0.3]+[0.6]\*2+[0.7]\*5+[0.8]\*2 | 
| DenseNet-40 | [0.]+[0.4]\*12+[0.2]+[0.5]\*12+[0.3]+[0.6]\*12 | 
| ResNet-50 | [0.1]+[0.2]\*3+[0.5]\*16 | 
| MobileNet-V1 | [0.]+[0.3]\*12 | 
| MobileNet-V2 | [0.]+[0.3]*7 | 

With the compression rate, our training module (evaluate_cifar.py and evaluate.py) can automatically calculate the params and FLOPs of that model and record them  in the training logger.

After training, a total of four files can be found in the `job_dir`, including best model, final model, config file and logger file.

### Experimental Results

We provide our pre-trained models, and pruned models. For your ease of reproducibility, the training loggers and configuration files are attached with the url link as well.

(The percentages in parentheses indicate the pruning rate)

##### CIFAR-10

| Architecture | Params        | Flops          |  Accuracy | Model                                              |
|:----------:|:-------------:|:--------------:|:--------:|:------------------------------------------------------------:|
| VGG-16-BN(Baseline)      | 14.98M(0.0%) | 313.73M(0.0%) | 93.96%   | [pre-trained](https://drive.google.com/open?id=1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE) |
| VGG-16-BN      | 2.76M(81.6%) | 131.17M(58.1%) | 93.73%(-0.23%)   | [pruned](https://drive.google.com/drive/folders/1iTfZt6bWN9RsoYYv9JHOia0EOEB5vxSp?usp=sharing) |
| VGG-16-BN      | 2.50M(83.3%) | 104.78M(66.6%) | 93.56%(-0.40%)    | [pruned](https://drive.google.com/drive/folders/1guvmJ97al7dDE7pQ2gcYMpG4ASQyu2rK?usp=sharing) |
| VGG-16-BN      | 1.90M(87.3%) | 66.95M(78.6%) | 93.10%(-0.86%)    | [pruned](https://drive.google.com/drive/folders/1NWssBVcGJs_d72B89A7vdIzhaDC0zvUX?usp=sharing) |
| ResNet-56(Baseline)   | 0.85M(0.0%) | 125.49M(0.0%) |  93.26%   |  [pre-trained](https://drive.google.com/open?id=1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T)
| ResNet-56   | 0.66M(22.3%) | 90.35M(28.0%) |  93.85%(+0.59%)   | [pruned](https://drive.google.com/drive/folders/1sfArXzP1iKtBjGMjXXL7GpcgNPjBjRjy?usp=sharing) |
| ResNet-56   | 0.48M(42.8%) | 65.94M(47.4%) | 93.57%(+0.31%)   | [pruned](https://drive.google.com/drive/folders/12Z21U0eUOQSRHde0Nk7TUwgt0i8gCpTm?usp=sharing) |
| ResNet-56   | 0.24M(70.0%) | 34.78M(74.1%) | 92.32%(-0.94%)  | [pruned](https://drive.google.com/drive/folders/1Pkyhi5PQRHTXE4eDMyJ3nYTNUIZYP5xy?usp=sharing) |
| ResNet-110(Baseline)   | 1.72M(0.0%) | 252.89M(0.0%) |  93.50%   |  [pre-trained](https://drive.google.com/open?id=1uENM3S5D_IKvXB26b1BFwMzUpkOoA26m)
| ResNet-110   | 1.04M(39.1%) | 140.54M(44.4%) |  94.20%(+0.70%)   | [pruned](https://drive.google.com/drive/folders/1Cci2so27VsEJRhwJ01HbN963L1tumB74?usp=sharing) |
| ResNet-110   | 0.72M(58.1%) | 101.97M(59.6%) |  93.81%(+0.31%)   | [pruned](https://drive.google.com/drive/folders/1poMhEDjWOn1UWjMkMz43ORVRdygxDg83?usp=sharing) |
| ResNet-110   | 0.54M(68.3%) | 71.69M(71.6%) |  93.23%(-0.27%)   | [pruned](https://drive.google.com/drive/folders/1pR6v1fC2tbzsXP_RqDe05Af8J42q1EgO?usp=sharing) |
| GoogLeNet(Baseline)  | 6.15M(0.0%) | 1520M(0.0%) | 95.05%   |   [pre-trained](https://drive.google.com/open?id=1rYMazSyMbWwkCGCLvofNKwl58W6mmg5c)
| GoogLeNet  | 2.85M(53.5%) | 649.19M(57.2%) | 95.04%(-0.01%)   | [pruned](https://drive.google.com/drive/folders/1fcoRYP3lxSXxBsZtjl8tEJIKZKebJhEC?usp=sharing) |
| GoogLeNet  | 2.09M(65.8%) | 395.42M(73.9%) | 94.82%(-0.23%)   | [pruned](https://drive.google.com/drive/folders/1QKs2yM0ApsrRr1Tya7kpXfc-B5a4PDXK?usp=sharing) |
| DenseNet-40(Baseline)  | 1.04M(0.0%) | 282.00M(0.0%) | 94.81%   | [pre-trained](https://drive.google.com/open?id=12rInJ0YpGwZd_k76jctQwrfzPubsfrZH)
| DenseNet-40  | 0.62M(40.1%) | 173.39M(38.5%) | 94.51%(-0.30%)   | [pruned](https://drive.google.com/drive/folders/1gCOD7MCyjqY7JYKD_WzznRQN-A85kEqk?usp=sharing) |
| DenseNet-40  | 0.45M(56.5%) | 133.17M(52.7%) | 93.91%(-0.90%)   | [pruned](https://drive.google.com/drive/folders/1s7iuIGKR19-z7fqL54BlszplMmojAP7s?usp=sharing) |
| DenseNet-40  | 0.39M(61.9%) | 113.08M(59.9%) | 93.66%(-1.21%)   | [pruned](https://drive.google.com/drive/folders/14bP40bwViUIy38z_x0isdLYnXIsO0S2H?usp=sharing) |


##### ImageNet (Lacking GPUs... Coming as soon as Possible)
| Architecture | Params        | Flops      | Use Dali | Lr Type | Top-1 Accuracy | Top-5 Accuracy | Model |
|:----------:|:-------------:|:--------------:|:--------:|:------------------:|:------------------:|:----------------------------:|:---:|
| ResNet-50(baseline)  |       25.55M(0.0%)          |      4.11B(0.0%)      |  | |   76.15%      |       92.87%         | [pre-trained](https://drive.google.com/open?id=1OYpVB84BMU0y-KU7PdEPhbHwODmFvPbB)|
| ResNet-50  |       15.09M(40.8%)          |      2.26B(44.8%)      |  No  | step |  75.56%      |       92.63%         | [pruned](https://drive.google.com/drive/folders/1vQsMbwxqtvWv44GE9Ni-ygn1eJNARn-_?usp=sharing)|
| ResNet-50  |       11.05M(56.7%)          |      1.52B(62.8%)      |  No | step |   74.19%      |       91.94%         | [pruned](https://drive.google.com/drive/folders/1h-HoeFsHWpIYs_49jzrSs9fDWyAA6iKp?usp=sharing)|
| ResNet-50  |       8.02M(68.6%)          |      0.95B(76.7%)      |  No |  step | 72.30%      |      90.74%         | [pruned](https://drive.google.com/drive/folders/1LldnR--CUp-tV1SRQ8AtlioBjgbzO6IV?usp=sharing)|
| MobileNet-v2(baseline)|      3.50M(0.0%)           |       314.13M(0.0%)     |   |  |    71.70%         |      90.43%          |  [pre-trained](https://drive.google.com/file/d/16YAmYG9u9NB6ztyzSz6e21qSKcr9AT6e/view?usp=sharing)  |


To verify our model performance, please use the command below (make sure you are using the corresponding compression rate in the configuration file of that model):
##### CIFAR-10:

```shell
python evaluate_cifar.py \
--data_dir [CIFAR-10 dataset dir] \
--job_dir ./result/[model name]/[folder name] \
--arch [model name](vgg_16_bn, resnet_56, resnet_110, googlenet, densenet_40) \
--test_only \
--test_model_dir [test model dir] \
--compress_rate [compress rate] \
--gpu [gpu_id]
```
##### ImageNet:
```
python evaluate.py \
--data_dir [ImageNet dataset dir] \
--job_dir ./result/[model name]/[folder name] \
--arch [model name](resnet_50, mobilenet_v1, mobilenet_v2) \
--test_only \
--test_model_dir [test model dir] \
--compress_rate [compress rate] \
--gpu [gpu_id]
```

