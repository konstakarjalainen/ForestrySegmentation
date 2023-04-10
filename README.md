# Bachelor's thesis by Konsta Karjalainen
Image segmentation for Forestry scenes

This repository has two codes for semnatic segmentation and instance segmentation. U_net folder and deeplab_fcn folder are for semantic segmentation and Mask R-CNN is for instance segmentation.

# Sematic Segmentation:
The code is based on U-NET implementation: https://github.com/milesial/Pytorch-UNet

A couple of changes in code were made to work it with datasets: http://deepscene.cs.uni-freiburg.de/ and http://rugd.vision/

# Instance Segmentation:
The code is based on logpiles segmentation implementation: https://github.com/norlab-ulaval/logpiles_segmentation

Used datasets are TimberSeg 1.0 used in the logpiles segmentation: https://data.mendeley.com/datasets/y5npsm3gkj/ and CanaTree100 dataset from: https://github.com/norlab-ulaval/PercepTreeV1

Also pretrained weights are used from the SynthTree43k training from PercepTreeV1 implentation.
