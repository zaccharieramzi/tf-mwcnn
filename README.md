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

# Contributing

I am welcoming any feedback under the form of GitHub issues or Pull requests.
