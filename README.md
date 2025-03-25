# Improved Vessel Segmentation with Symmetric Rotation-Equivariant U-Net

[![PyPI version](https://img.shields.io/pypi/v/SRE-Conv.svg)](https://pypi.org/project/SRE-Conv/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

*Jiazhen Zhang, Yuexi Du, Nicha C. Dvornek, John A. Onofrey*

*Yale University*

## Abstract

> Automated segmentation plays a pivotal role in medical image analysis and computer-assisted interventions. Despite the promising performance of existing methods based on convolutional neural networks (CNNs), they neglect useful equivariant properties for images, such as rotational and reflection equivariance.  This limitation can decrease performance and lead to inconsistent predictions, especially in applications like vessel segmentation where explicit orientation is absent. While existing equivariant learning approaches attempt to mitigate these issues, they substantially increase learning cost, model size, or both. To overcome these challenges, we propose a novel application of an efficient symmetric rotation-equivariant (SRE) convolutional (SRE-Conv) kernel implementation to the U-Net architecture, to learn rotation- and reflection-equivariant features, while also reducing the model size dramatically. We validate the effectiveness of our method through improved segmentation performance on retina vessel fundus imaging. Our proposed SRE U-Net not only significantly surpasses standard U-Net in handling rotated images, but also outperforms existing equivariant learning methods and does so with a reduced number of trainable parameters and smaller memory cost.

## SRE-Conv Segmentation

To install the SRE-Conv, please refer to [SRE-Conv](https://github.com/XYPB/SRE-Conv).

To perform the SRE-Conv Segmentation, please refer to `./src/segm_lightning_retina_SRE_Conv.ipynb`.

## Reference
    @article{zhang2025improved,
      title={Improved Vessel Segmentation with Symmetric Rotation-Equivariant U-Net},
      author={Zhang, Jiazhen and Du, Yuexi and Dvornek, Nicha C and Onofrey, John A},
      journal={arXiv preprint arXiv:2501.14592},
      year={2025}
    }

