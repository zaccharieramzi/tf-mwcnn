# TensorFlow implementation of the MWCNN

[![Build Status](https://travis-ci.com/zaccharieramzi/tf-mwcnn.svg?branch=master)](https://travis-ci.com/zaccharieramzi/tf-mwcnn)

The MWCNN is a network introduced by Pengju Liu et al. in
"Multi-Level Wavelet-CNN for Image Restoration", CVPR 2018, and refined in
"Multi-Level Wavelet Convolutional Neural Networks", IEEE Access June 2019.
If you use this network, please cite their work appropriately.

The official implementation is available [here](https://github.com/lpj0/MWCNN)
in Matlab and [here](https://github.com/lpj0/MWCNN_PyTorch) in Pytorch.
A second version is available [here](https://github.com/lpj-github-io/MWCNNv2)
in Pytorch.

The goal of this implementation in TensorFlow is to be easy to read and to adapt:
- all the code is in one file
- defaults are those from the paper
- there is no other imports than from TensorFlow

The only thing I am currently not happy about is the implementation of `IWT`.
Currently, it is very difficult to read because tensor slice assignment is
impossible as-is in TensorFlow.
See for example this [SO question](https://stackoverflow.com/q/62092147/4332585).

# Differences between 2018 and 2019

There are some differences in the 2 papers defining the MWCNN.
We provide the 2 defaults and the ways to implement both slightly different
architectures.

The Figures defining the implementations are:
- 3. in the conference paper.
- 3. in the journal paper.

## Convolution blocks

The batch normalization is the only difference.
It is present in the 2018 conference paper.
It is absent in the 2019 journal paper.

In the 2018 conference paper:
> Each layer of the CNN block is composed of convolution with 3 × 3 filters
> (Conv), batch normalization (BN), and rectified linear unit (ReLU)
> operations.

In the 2019 journal paper:
> More concretely, each layer contains convolution with 3×3 filters (Conv), and
> rectified linear unit (ReLU) operations.

## Scales

In the conference paper, there is a DWT operation as the first operation
in the network.
In the journal paper, there are first convolutions before the DWT.

## Layers

In the conference paper, there are 4 convolution blocks per scale.
In the journal paper, there are 3 convolution blocks per scale.

# Contributing

I am welcoming any feedback under the form of GitHub issues or Pull requests.
