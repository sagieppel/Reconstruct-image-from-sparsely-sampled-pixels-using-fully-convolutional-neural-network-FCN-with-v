# Reconstructing image from sparsely sampled pixels using fully convolutional neural network (FCN) with valve filters
Take image in which only fraction of the pixels are sampled and reconstruct/upsample the full image using fully convolutional neural nets and valve filters (Tensorflow implementation).
![](/Image1.png)  ![](/ValveFilterScheme.png)


## Valve filter method for reconstructing image from sparsely sampled input image 
Recent works showed that upsampling of images using deep neural nets give state of the art results. These nets used fully convolutional neural nets that take small images and upsample/rescale it  to larger size while using learned features to fill the missing information. 
However, a different kind of upsampling/reconstruction is needed in cases where full scale image is already available but only small subset of of the pixels in the image are determined in sparse unorganized positions (Figure). The ability to reconstruct image from sparsely sampled point represent challenging problem for convolutional neural nets, since such nets is based on learned filters (masks) that convolve with the image to detect various of patterns. However with only small fraction of the pixels in the image sampled (and the rest are zero) such filters can give an unreliable response. For example edge filter such as sobel cannot give a reliable response in regions where only single pixel is sampled, nonetheless it can give a strong response if the value of this single pixel is high. To tackle this problem this code use  valve filter that act to regulate the activation of the filters based on the arrangement of sample pixels in their region and the confidence in the filter response. The schematic of the method is shown in the figure. For each filter that act on the image a corresponding valve filter exist. This valve filter act on the binary image that  contain the arrangement as sampled pixel (value of 1 in each sampled pixel position  and zero elsewhere). The valve filters generate confidence map correspond to the activation of each corresponding image filter. The confidence and the activation map are then multiplied element wise to give the normalized activation map which is then applied to fully convolutional neural net for the image reconstruction.  In this method filter that have strong response but low confidence will end up with low activation in the normalized activation map.
## Instructions:
### Training: 
In train.py:
1) Set folder with  images in: Train_Image_Dir
2) Set Images size in: Im_Width and Im_Hight
3)Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate
4)Run Script the trained net will appear in the log_dir 
 
### Using trained net to reconstruct sparsely sample image:
In: RunPrediction.py
 
Assume that you already have trained model in log_dir, if you dont have trained model see training section
1) Set folder with  images in: Image_Dir
2) Set Sampling rate (hence fraction of pixels) sampled from each 3) image in: SamplingRate.
4) Set the directory in which you want the output image to appear to: OUTPUT_Dir
 
 
The code was run on python 2.7 using tensorflow 1.1
This code is based on VGG19 Fully convolutional neural nets supply in: https://github.com/shekkizh/FCN.tensorflow
By Sarath Shekkizhar with mit licence


