#Script for training network for recosttucting image from sparsly sample pixels
#Hence take image in which only small fraction of the pixels are known and reconstruct the full image
#The unknown pixels are marked as 0

#Instructions for training
#Set folder with  images in: Train_Image_Dir
#Set Images size in: Im_Width and Im_Hight
#Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate
#Run training: the train model will appear in the log file
#To use the trained network for reconstructing image from sparsely sampled pixels see: RunPrediction.py

from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import Build_Net as BuildNet #Were the net is built
import ImageReader as ImageReader # loader for images
import os
import scipy.misc as misc
import random

Train_Image_Dir="/media/sagi/1TB/Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/"# Directory with image to train
SamplingRate=0.1 # Fraction of pixels to be sampled from image for training
Im_Width=400 #Width and hight to which all the images will be resized
Im_Hight=400


Batch_Size=2 # For training (number of images trained per iteration)
logs_dir="logs/" # Were the trained model and all output will be put
learning_rate=1e-5#Learning rate for Adam Optimizer
Vgg_Model_Dir="Model_zoo/" #Directory of the pretrained VGG model if model not there it will be automatically download
TrainLossTxtFile=logs_dir+"TrainLoss.txt"#



MAX_ITERATION = int(60000) #Maximal training iteration

if not os.path.exists(logs_dir): os.makedirs(logs_dir)

###########################################Solver for the net training################################################################################################################
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

################################################################################################################################################################################
def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty") #Dropout probability
    Sparse_Sampled_Image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_Sparse_image") #Input image sparsly sampled image
    Full_Image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="Full_image")  # Full image all pixels filled
    Binary_Point_Map = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="Binary_Point_Map") # Binary image with all the sample point marked as 1 and the rest of the pixels are 0, hence binary map of sampled point
    ReconstructImage = BuildNet.inference(Sparse_Sampled_Image,Binary_Point_Map,keep_probability,3,Vgg_Model_Dir )# Here the graph(net) is builded
    loss =tf.reduce_mean(tf.abs(ReconstructImage - Full_Image, name="L1_Loss"))  # Define loss function for training as the difference between reconstruct image and ground truth image


    # tf.summary.scalar("L1_Loss", loss)

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    #print("Setting up summary op...")
    #summary_op = tf.summary.merge_all()

    print("Reading images list")
    TrainImages=[]   #Train Image List

    TrainImages += [each for each in os.listdir(Train_Image_Dir) if each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith( '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith('.tif') or each.endswith('.gif')]  # Get list of training images


    print('Number of  Train images='+str(len(TrainImages)))

#-------------------------Training Region-----------------------------------------------------------------------------------------------------------------------------

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
#---------------------------Start Training: Create loss files----------------------------------------------------------------------------------------------------------
    Nimg=0
    f = open(TrainLossTxtFile, "w")# Create text file for writing the loss trough out the training
    f.write("Iteration\tTrain_Loss\t Learning Rate="+str(learning_rate))
    f.close()
#-----------------------------------------------------------------------------------------------------------------
    Epoch=0
#..............Start Training loop: Main Training....................................................................
    for itr in range(1,MAX_ITERATION+1):
        if Nimg>=len(TrainImages)-1: # End of an epoch
           Nimg=0
           random.shuffle(TrainImages)  #Suffle images every epoch
           Epoch += 1
           print("Epoch "+str(Epoch)+" Completed")
#.....................Load images for training
        batch_size=np.min([Batch_Size,len(TrainImages)-Nimg])
        FullImages = np.zeros([batch_size,Im_Hight,Im_Width,3], dtype=np.int)
        SparseSampledImages = np.zeros([batch_size,Im_Hight,Im_Width,3], dtype=np.int)
        BinarySamplesMap = np.zeros([batch_size, Im_Hight, Im_Width, 1], dtype=np.int)
        for fi in range(batch_size):
            FullImages[fi],SparseSampledImages[fi],BinarySamplesMap[fi]=ImageReader.LoadImages(Train_Image_Dir +TrainImages[Nimg],Im_Hight,Im_Width,SamplingRate)
            Nimg+=1


#.......................Run one batch of training...............................................................................
        feed_dict = {Sparse_Sampled_Image: SparseSampledImages,Binary_Point_Map:BinarySamplesMap, Full_Image: FullImages, keep_probability: 0.6+np.random.rand()*0.4}# Run one cycle of traning
        sess.run(train_op, feed_dict=feed_dict)
#......................Write training set loss..........................................................................
        if itr % 10==0:
            feed_dict = {Sparse_Sampled_Image: SparseSampledImages,Binary_Point_Map:BinarySamplesMap, Full_Image: FullImages,keep_probability: 1}
            train_loss= sess.run(loss, feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g " % (itr, train_loss))
          #  summary_writer.add_summary(summary_str, itr)
            with open(TrainLossTxtFile, "a") as f:#Write training loss for file
                 f.write("\n"+str(itr)+"\t"+str(train_loss))
                 f.close()
#....................Save Trained net (ones every 1000 training cycles...............................................
        if itr%200==0 :
            print("Saving Model")
            saver.save(sess, logs_dir + "model.ckpt", itr)# save trained model
print("Finished Running")
if __name__ == "__main__":
    tf.app.run()
