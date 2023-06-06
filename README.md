# tts-trade-predict (CBAM.PyTorch)
Non-official implement of Paper：CBAM: Convolutional Block Attention Module

## Introduction
The codes are [PyTorch](https://pytorch.org/) re-implement version for paper: CBAM: Convolutional Block Attention Module

> Woo S, Park J, Lee J Y, et al. CBAM: Convolutional Block Attention Module[J]. 2018. [ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

## Structure

The overview of CBAM. The module has two sequential sub-modules:
channel and spatial. The intermediate feature map is adaptively refined through
our module (CBAM) at every convolutional block of deep networks.