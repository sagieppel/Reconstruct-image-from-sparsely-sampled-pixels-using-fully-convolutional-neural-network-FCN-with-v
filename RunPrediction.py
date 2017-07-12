#Script for training network for recosttucting image from sparsly sample pixels
#Hence take image in which only small fraction of the pixels are known and reconstruct the full image
#The unknown pixels are marked as 0

#Instructions for Running Prediction
#Assume that you already have trained model in log_dir, if you dont have trained model see: trained.py for training
#Set folder with  images in: Image_Dir
#Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate
#If your image are already sampled change the reader to read the binary sampling map and the sample image
#Set the directory in which you want the output image to appear to: OUTPUT_Dir
#Run

from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import Build_Net as BuildNet #Were the net is built
import ImageReader as ImageReader # loader for images
import os
import scipy.misc as misc
import random

Image_Dir="/media/sagi/1TB/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/validation/"# Input dir with image for prediction
OUTPUT_Dir="/home/sagi/Desktop/OutputDir/"# Output directory
SamplingRate=0.1 # Fraction of pixels to be sampled from image (assume the input image are not already sampled)

logs_dir="logs/" # Were the trained model and all output will be saved
Vgg_Model_Dir="Model_zoo/" #Directory of the pretrained VGG model if model not there it will be automatically download



if not os.path.exists(OUTPUT_Dir): os.makedirs(OUTPUT_Dir)


################################################################################################################################################################################
def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty") #Dropout probability
    Sparse_Sampled_Image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_Sparse_image") #Input image sparsly sampled image
    Binary_Point_Map = tf.placeholder(tf.float32, shape=[None, None, None, 1],name="Binary_Point_Map")  # Binary image with all the sample point marked as 1 and the rest of the pixels are 0, hence binary map of sampled point
    ReconstructImage = BuildNet.inference(Sparse_Sampled_Image, Binary_Point_Map, keep_probability, 3,Vgg_Model_Dir)  # Here the graph(net) is builded

    print("Reading images list")
#---------------------Read list of image for recostruction------------------------------------------------------------
    Images=[]   #Train Image List

    Images += [each for each in os.listdir(Image_Dir) if each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith( '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith('.tif') or each.endswith('.gif')]  # Get list of training images


    print('Number of images='+str(len(Images)))

#-------------------------Load trained mode----------------------------------------------------------------------------------------------------------------------------

    sess = tf.Session() #Start Tensorflow session

    print("Setting up Saver...")
    saver = tf.train.Saver()
   # summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if trained model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("Error no trained model found in log dir "+logs_dir+"  For creating trained model see: Train.py")
        return

    SumLoss=0
#..............Start image reconstruction....................................................................
    for itr in range(len(Images)):
#.....................Load images for prediction-------------------------------------
        print(str(itr)+") Reconstructing: "+Image_Dir +Images[itr])

        FullImage,SparseSampledImage,BinarySamplesMap=ImageReader.LoadImages(Image_Dir +Images[itr],0,0,SamplingRate)

#.......................Run one  prediction...............................................................................
        feed_dict = {Sparse_Sampled_Image: SparseSampledImage,Binary_Point_Map:BinarySamplesMap, keep_probability: 1}# Run one cycle of traning
        ReconImage=sess.run(ReconstructImage, feed_dict=feed_dict)# run image reconstruction using network
#......................Save image..........................................................................
        #ReconImage[ReconImage>255]=255
        #ReconImage[ReconImage<0]=0
        loss=np.mean(np.abs(ReconImage[0]-FullImage[0]))
        SumLoss+=loss
        print("Loss="+str(loss)+"        Mean loss="+str(SumLoss/(itr+1)))
        misc.imsave(OUTPUT_Dir+"/"+Images[itr][0:-4]+"_Reconstructed"+Images[itr][-4:],ReconImage[0])
        misc.imsave(OUTPUT_Dir + "/" + Images[itr][0:-4] + "_Original" + Images[itr][-4:], FullImage[0])
        misc.imsave(OUTPUT_Dir + "/" + Images[itr][0:-4] + "_Sampled" + Images[itr][-4:], SparseSampledImage[0])
        misc.imsave(OUTPUT_Dir + "/" + Images[itr][0:-4] + "_BinaryMap" + Images[itr][-4:], BinarySamplesMap[0,:,:,0])

print("Finished Running")
if __name__ == "__main__":
    tf.app.run()
