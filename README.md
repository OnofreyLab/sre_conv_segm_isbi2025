# SRE-Conv: Symmetric Rotation Equivariant Convolution for Biomedical Image Classification

This is the official implementation of paper "Improved Vessel Segmentation with Symmetric Rotation-Equivariant U-Net" (accepted by **ISBI 2025**)

*Jiazhen Zhang, Yuexi Du, Nicha C. Dvornek, John A. Onofrey*

*Yale University*

## Abstract

> Automated segmentation plays a pivotal role in medical image analysis and computer-assisted interventions. Despite the promising performance of existing methods based on convolutional neural networks (CNNs), they neglect useful equivariant properties for images, such as rotational and reflection equivariance.  This limitation can decrease performance and lead to inconsistent predictions, especially in applications like vessel segmentation where explicit orientation is absent. While existing equivariant learning approaches attempt to mitigate these issues, they substantially increase learning cost, model size, or both. To overcome these challenges, we propose a novel application of an efficient symmetric rotation-equivariant (SRE) convolutional (SRE-Conv) kernel implementation to the U-Net architecture, to learn rotation- and reflection-equivariant features, while also reducing the model size dramatically. We validate the effectiveness of our method through improved segmentation performance on retina vessel fundus imaging. Our proposed SRE U-Net not only significantly surpasses standard U-Net in handling rotated images, but also outperforms existing equivariant learning methods and does so with a reduced number of trainable parameters and smaller memory cost.


## Installation of SRE-Conv

Please see details at: [SRE-Conv](https://github.com/XYPB/SRE-Conv).

We provide both the PyPI package for SRE-Conv and the code to reproduce the experiment results in this repo

To install and directly use the SRE-Conv, please run the following command:
```bash
pip install SRE-Conv
```

The minimal requirement for the SRE-Conv is:
```bash
"scipy>=1.9.0",
"numpy>=1.22.0",
"torch>=1.8.0"
```

**Note**: Using lower version of torch and numpy should be fine given that we didn't use any new feature in the new torch version, but we do suggest you to follow the required dependencies. If you have to use the different version of torch/numpy, you may also try to install the package from source code at 

## Usage
```python
>>> import torch
>>> from SRE_Conv import SRE_Conv2d, sre_resnet18
>>> x = torch.randn(2, 3, 32, 32)
>>> sre_conv = SRE_Conv2d(3, 16, 3)
>>> conv_out = SRE_conv(x)
>>> sre_r18 = sre_resnet18()
>>> output = sre_r18(x)
```

## Train & Evaluation on MedMNIST

To reproduce the experiment results, you may also need to install the following packages:
```bash
"medmnist>=3.0.0"
"grad-cam>=1.5.0"
"matplotlib"
"imageio"
```

Run the following comment to train the model and evaluate the performance under both flip and rotation evaluation.
```bash
cd ./src
# 2D MedMNIST
python main.py --med-mnist <medmnist_dataset> --epochs 100 --model-type sre_resnet18 --sre-conv-size-list 9 9 5 5 -b 128 --lr 2e-2 --cos --sgd --eval-rot --eval-flip --train-flip-p 0 --log --cudnn --moco-aug --translate-ratio 0.1 --translation --save-model  --save-best --res-keep-conv1
# 3D MedMNIST
python main.py --med-mnist <medmnist3d_dataset> --epochs 100 --model-type sre_r3d_18 --ri-conv-size-list 5 5 5 5 -b 4 --lr 1e-2 --cos --sgd --eval-rot --res-keep-conv1 --log --cudnn --moco-aug
```


## Reference

*We will update this soon...*
