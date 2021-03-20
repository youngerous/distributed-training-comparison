# Distributed Training in PyTorch

There are some distributed training steps you can try according to [PyTorch Document](https://pytorch.org/tutorials/beginner/dist_overview.html).


> PyTorch provides several options for data-parallel training. For applications that gradually grow from simple to complex and from prototype to production, the common development trajectory would be:
> 1. Use **single-device** training, if the data and model can fit in one GPU, and the training speed is not a concern.
> 2. Use **single-machine multi-GPU DataParallel**, if there are multiple GPUs on the server, and you would like to speed up training with the minimum code change.
Use single-machine multi-GPU DistributedDataParallel, if you would like to further speed up training and are willing to write a little more code to set it up.
> 3. Use **multi-machine DistributedDataParallel** and the launching script, if the application needs to scale across machine boundaries.
> 4. Use torchelastic to launch distributed training, if errors (e.g., OOM) are expected or if the resources can join and leave dynamically during the training.


In this repo, I compared **single-device(1)** with **single-machine multi-GPU DataParallel(2)** and **single-machine multi-GPU DistributedDataParallel**.

## Environment
- Nvidia RTX 2080ti * 2
- torch==1.7.1
- torchvision==0.8.2

All dependencies are written in [requirements.txt](https://github.com/youngerous/distributed-training-comparison/blob/main/requirements.txt), and you can also access through [Dockerfile](https://github.com/youngerous/distributed-training-comparison/blob/main/Dockerfile).

## How to Run
All three folders - ```src/single/```, ```src/dp/```, and ```src/ddp/``` - are independent structures.

### Single
```sh
$ sh src/single/run_single.sh
```
### DataParallel
```sh
$ sh src/dp/run_dp.sh
```
### DistributedDataParallel
```sh
$ sh src/ddp/run_ddp.sh
```

## Result
Batch size is set to 128 or 256. It is recommended to use [SyncBatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html) in DDP training, but I used vanila BatchNorm so just trained on 256 batch size. Best model is selected according to validation top-1 accuracy.


And I did not care detailed hyperparameter settings, so you can change some settings in order to improve performance (e.g. using ADAM optimizer).

|  Dataset  |   Model   | Test Loss  | Top-1 Acc  | Top-5 Acc  | Batch Size |            Method             |
| :-------: | :-------: | :--------: | :--------: | :--------: | :--------: | :---------------------------: |
| CIFAR-100 | ResNet-18 |   1.3728   |   70.99%   |   91.57%   |    128     |            Single             |
| CIFAR-100 | ResNet-18 |   1.3394   |   70.64%   |   91.60%   |    256     |            Single             |
| CIFAR-100 | ResNet-18 |   1.2974   | **71.48%** |   91.65%   |    128     |       DataParallel (DP)       |
| CIFAR-100 | ResNet-18 |   1.3373   |   71.20%   |   91.53%   |    256     |       DataParallel (DP)       |
| CIFAR-100 | ResNet-18 | **1.2268** |   71.17%   | **91.84%** |    256     | DistributedDataParallel (DDP) |

- Experiment results are averaged value of random seed 2, 4, 42.
- Automatic Mixed Precision(AMP) is applied to every experiment.

## Reference
- [[Docs] Distributed Communication Package - torch.distributed](https://pytorch.org/docs/stable/distributed.html#)
- [[Post] Technologies behind Distributed Deep Learning - AllReduce :: Keisuke Fukuda](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/)
- [[Post] PyTorch Distributed Training :: leimao blog](https://leimao.github.io/blog/PyTorch-Distributed-Training/)
- [[Post] Distributed data parallel training in Pytorch :: yangkky blog](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)
- [[Repo] PyTorch Official Example](https://github.com/pytorch/examples/blob/master/imagenet/main.py)
- [[Repo] pytorch-distributed :: tczhangzhi](https://github.com/tczhangzhi/pytorch-distributed)
