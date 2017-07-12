from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils


#########################Load Weigths function############################################################
def loadWeights(i,weights,LayerName):
    kernels, bias = weights[i][0][0][0][0]
    # matconvnet: weights are [width, height, in_channels, out_channels]
    # tensorflow: weights are [height, width, in_channels, out_channels]

    kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)),name=LayerName+"_w")  # Transform the np weight matrix to tf matrix that will be used, as the conv layer shape and intial weights, Set the shape of convlutional layer based on the weight matrix and load it as tf matrix,
    bias = utils.get_variable(bias.reshape(-1), name=LayerName+"_b")  # Transform biase matrix to tensorflow arrays
    return kernels, bias
#################################Load Weight for one layer input (second input)##################################################################################
def loadWeightsFlat(i,weights,LayerName):
    kernels, bias = weights[i][0][0][0][0]
    kernels=kernels[:, :, 1, :] # The original layer is for input depth 3 this is for input depth of 1
    kernels = np.expand_dims(kernels, axis=2)
    # matconvnet: weights are [width, height, in_channels, out_channels]
    # tensorflow: weights are [height, width, in_channels, out_channels]
    kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)),name=LayerName+"_w")  # Transform the np weight matrix to tf matrix that will be used, as the conv layer shape and intial weights, Set the shape of convlutional layer based on the weight matrix and load it as tf matrix,
    bias = utils.get_variable(bias.reshape(-1), name=LayerName+"_b")  # Transform biase matrix to tensorflow arrays
    return kernels, bias


###################################Build VGG Encoder with additional input#################################################################################
##########################VGG Build VGG encoder layer by layer##################################################################
def Build_vgg_net(weights, Sampled_image,Binary_Point_Map): #Build and load vgg net from weights and input image, not that weight define shape of convolution as well as they weight if you want to change this you might want to change this


    net = {}  # Dictionary that will contain all layers associate with their names (note the names are layers name and layers are tf layers)
    #***************************LAYER 1**********************************************************************************************************
    #-------------------------------Conv1_1---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(0,weights,"conv1_1_img")
    #current = utils.conv2d_basic(current, kernels,bias)  # set Conv layer  with loaded weights and biases note that the current layer is both input and ouput
    conv1_1_img = tf.nn.bias_add(tf.nn.conv2d(Sampled_image, kernels, strides=[1, 1, 1, 1],padding="SAME"),bias)  # Padding same mean the output is same size as input?
    relu1_1_img = tf.nn.relu(conv1_1_img, name="relu1_1_img")
    #------------------------Conv1_1_b For Second Input---------------------------------------------------------------------
    kernels, bias = loadWeightsFlat(0, weights, "conv1_1_Valve")
    # current = utils.conv2d_basic(current, kernels,bias)  # set Conv layer  with loaded weights and biases note that the current layer is both input and ouput
    conv1_1_Valve = tf.nn.bias_add(tf.nn.conv2d(Binary_Point_Map, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    relu1_1_Valve = tf.nn.relu(conv1_1_Valve, name="relu1_1_Valve")
    relu1_1=relu1_1_Valve*relu1_1_img #multiply response of valve filter in response of image filter
    #conv1_1=conv1_1_b+conv1_1_a
    #conv1_1=tf.add(conv1_1_b,conv1_1_a,name="conv1_1")

    #conv1_1 = conv1_1_b * conv1_1_a
    #conv1_1=tf.multiply(conv1_1_b,conv1_1_a,name="conv1_1")
    #net["conv1_1"] = conv1_1
    #--------------------------Relu1_1----------------------------------------------------------------------------------------------------------------
  #  if FLAGS.debug: utils.add_activation_summary(conv1_1)
    net["relu1_1"] = relu1_1  # load layer to the net dictionary according to its name (this not essential but make it easy to extract specific layer for future mendling)
    # -------------------------------Conv1_2---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(2, weights,"conv1_2")
    conv1_2 = tf.nn.bias_add(tf.nn.conv2d(relu1_1, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv1_2"] = conv1_2
    # --------------------------Relu1_1----------------------------------------------------------------------------------------------------------------
    relu1_2 = tf.nn.relu(conv1_2, name="relu1_2")
    net["relu1_2"] = relu1_1
    #--------------------------- Pool 1----------------------------------------------------------------------------------------------------------
    pool1=tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    net["pool1"] = pool1
    #-----------------------------------------------------------------------------------------------------------------------------------------



   ## ***************************LAYER 2**********************************************************************************************************
    # -------------------------------Conv2_1---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(5, weights,"conv2_1")
    conv2_1 = tf.nn.bias_add(tf.nn.conv2d(pool1, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv2_1"] = conv2_1
    # --------------------------Relu2_1----------------------------------------------------------------------------------------------------------------
    relu2_1 = tf.nn.relu(conv2_1, name="relu2_1")
   # if FLAGS.debug: utils.add_activation_summary(conv1_1)
    net["relu2_1"] = relu2_1  # load layer to the net dictionary according to its name (this not essential but make it easy to extract specific layer for future mendling)
    # -------------------------------Conv2_2---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(7, weights,"conv2_2")
    conv2_2 = tf.nn.bias_add(tf.nn.conv2d(relu2_1, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv2_2"] = conv2_2
    # --------------------------Relu2_2----------------------------------------------------------------------------------------------------------------
    relu2_2 = tf.nn.relu(conv2_2, name="relu2_2")
    net["relu2_2"] = relu2_2
    # --------------------------- Pool 1----------------------------------------------------------------------------------------------------------
    pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    net["pool2"] = pool2
    #-----------------------------------------------------------------------------------------------------------------------------------------

 ##***************************LAYER 3**********************************************************************************************************
    # -------------------------------Conv3_1---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(10, weights,"conv3_1")
    conv3_1 = tf.nn.bias_add(tf.nn.conv2d(pool2, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv3_1"] = conv3_1
    # --------------------------Relu3_1----------------------------------------------------------------------------------------------------------------
    relu3_1 = tf.nn.relu(conv3_1, name="relu3_1")
    net["relu3_1"] = relu3_1  # load layer to the net dictionary according to its name (this not essential but make it easy to extract specific layer for future mendling)
    # -------------------------------Conv3_2---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(12, weights,"conv3_2")
    conv3_2 = tf.nn.bias_add(tf.nn.conv2d(relu3_1, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv3_2"] = conv3_2
    # --------------------------Relu3_2----------------------------------------------------------------------------------------------------------------
    relu3_2 = tf.nn.relu(conv3_2, name="relu3_2")
    net["relu3_2"] = relu3_2
    # -------------------------------Conv3_3---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(14, weights,"conv3_3")
    conv3_3 = tf.nn.bias_add(tf.nn.conv2d(relu3_2, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv3_3"] = conv3_3
    # --------------------------Relu3_3---------------------------------------------------------------------------------------------------------------
    relu3_3 = tf.nn.relu(conv3_3, name="relu3_3")
    net["relu3_3"] = relu3_3
    # -------------------------------Conv3_4---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(16, weights,"conv3_4")
    conv3_4 = tf.nn.bias_add(tf.nn.conv2d(relu3_3, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv3_4"] = conv3_4
    # --------------------------Relu3_4---------------------------------------------------------------------------------------------------------------
    relu3_4 = tf.nn.relu(conv3_4, name="relu3_4")
    net["relu3_4"] = relu3_4
    # --------------------------- Pool 1----------------------------------------------------------------------------------------------------------
    pool3 = tf.nn.max_pool(relu3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    net["pool3"] = pool3
    # -----------------------------------------------------------------------------------------------------------------------------------------


##***************************LAYER 4**********************************************************************************************************
    # -------------------------------Conv4_1---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(19, weights,"conv4_1")
    conv4_1 = tf.nn.bias_add(tf.nn.conv2d(pool3, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv4_1"] = conv4_1
    # --------------------------Relu4_1----------------------------------------------------------------------------------------------------------------
    relu4_1 = tf.nn.relu(conv4_1, name="relu4_1")
    net["relu4_1"] = relu4_1  # load layer to the net dictionary according to its name (this not essential but make it easy to extract specific layer for future mendling)
    # -------------------------------Conv4_2---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(21, weights,"conv4_2")
    conv4_2 = tf.nn.bias_add(tf.nn.conv2d(relu4_1, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv4_2"] = conv4_2
    # --------------------------Relu4_2----------------------------------------------------------------------------------------------------------------
    relu4_2 = tf.nn.relu(conv4_2, name="relu4_2")
    net["relu4_2"] = relu4_2
    # -------------------------------Conv4_3---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(23, weights,"conv4_3")
    conv4_3 = tf.nn.bias_add(tf.nn.conv2d(relu4_2, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv4_3"] = conv4_3
    # --------------------------Relu4_3---------------------------------------------------------------------------------------------------------------
    relu4_3 = tf.nn.relu(conv4_3, name="relu4_3")
    net["relu4_3"] = relu4_3
    # -------------------------------Conv4_4---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(25, weights,"conv4_4")
    conv4_4 = tf.nn.bias_add(tf.nn.conv2d(relu4_3, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv4_4"] = conv4_4
    # --------------------------Relu4_4---------------------------------------------------------------------------------------------------------------
    relu4_4 = tf.nn.relu(conv4_4, name="relu4_4")
    net["relu4_4"] = relu4_4
    # --------------------------- Pool 1----------------------------------------------------------------------------------------------------------
    pool4 = tf.nn.max_pool(relu4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    net["pool4"] = pool4
    # -----------------------------------------------------------------------------------------------------------------------------------------


##***************************LAYER 5**********************************************************************************************************
    # -------------------------------Conv5_1---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(28, weights,"conv5_1")
    conv5_1 = tf.nn.bias_add(tf.nn.conv2d(pool4, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv5_1"] = conv5_1
    # --------------------------Relu5_1----------------------------------------------------------------------------------------------------------------
    relu5_1 = tf.nn.relu(conv5_1, name="relu5_1")
    net["relu5_1"] = relu5_1  # load layer to the net dictionary according to its name (this not essential but make it easy to extract specific layer for future mendling)
    # -------------------------------Conv5_2---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(30, weights,"conv5_2")
    conv5_2 = tf.nn.bias_add(tf.nn.conv2d(relu5_1, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv5_2"] = conv5_2
    # --------------------------Relu5_2----------------------------------------------------------------------------------------------------------------
    relu5_2 = tf.nn.relu(conv5_2, name="relu5_2")
    net["relu5_2"] = relu5_2
    # -------------------------------Conv5_3---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(32, weights,"conv5_3")
    conv5_3 = tf.nn.bias_add(tf.nn.conv2d(relu5_2, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv5_3"] = conv5_3
    # --------------------------Relu5_3---------------------------------------------------------------------------------------------------------------
    relu5_3 = tf.nn.relu(conv5_3, name="relu5_3")
    net["relu5_3"] = relu5_3
    # -------------------------------Conv5_4---------------------------------------------------------------------------------------------------------
    kernels, bias = loadWeights(34, weights,"conv5_4")
    conv5_4 = tf.nn.bias_add(tf.nn.conv2d(relu5_3, kernels, strides=[1, 1, 1, 1], padding="SAME"),bias)  # Padding same mean the output is same size as input?
    net["conv5_4"] = conv5_4
    # --------------------------Relu5_3---------------------------------------------------------------------------------------------------------------
    relu5_4 = tf.nn.relu(conv5_4, name="relu5_4")
    net["relu5_4"] = relu5_4
    #--------------------------------------------------------------------------------------------------------------------------------------------


    return net #Return array with all
###########################################################################################################################################################
def inference(Sampled_image,Binary_Point_Map, keep_prob, Num_Channels,model_dir):
    # Build network and load initial weights
    #image: tf  tensor of the input image
    #keep_prob: Probabality for dropout only applied during training
    """
    
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
    model_data = utils.get_model_data(model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]# Seem like the mean for every pixel in the image (float matrix in image size)
    mean_pixel = np.mean(mean, axis=(0, 1))# Seem like the mean rgb (float array length 3)

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(Sampled_image, mean_pixel)# Substract mean from every pixel, nothing more

    with tf.variable_scope("inference"):
        #image_net =Build_vgg_net_using_loop(weights, processed_image)
        image_net = Build_vgg_net(weights, processed_image,Binary_Point_Map) #This is were the encoder i.e the VGG net is built and loaded from pretrained  VGG net
        conv_final_layer = image_net["conv5_3"] #Continue the decoder from the last layer of the vgg encoder

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6") #Create tf weight for the new layer with initial weights with normal random distrubution mean zero and std 0.02
        b6 = utils.bias_variable([4096], name="b6") #Create tf biasefor the new layer with initial weights of 0
        conv6 = utils.conv2d_basic(pool5, W6, b6) #  Check the size of this net input is it same as input or is it 1X1
        relu6 = tf.nn.relu(conv6, name="relu6")
        #if FLAGS.debug: utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob) # Apply dropout for traning need to be added only for training

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7") #1X1 Convloution
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7) #1X1 Convloution
        relu7 = tf.nn.relu(conv7, name="relu7")
        #if FLAGS.debug: utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob) # Another dropout need to be used only for training

        W8 = utils.weight_variable([1, 1, 4096, 100], name="W8") # Basically the output num of classes imply the output is already the prediction this is flexible can be change however in multinet class number of 2 give good results
        b8 = utils.bias_variable([100], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape() # Set the output shape for the the transpose convolution output take only the depth since the transpose convolution will have to have the same depth for output
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, 100], name="W_t1") # Deconvolution/transpose in size 4X4 note that the output shape is of  depth NUM_OF_CLASSES this is not necessary in will need to be fixed if you only have 2 catagories
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"])) # Use strided convolution to double layer size (depth is the depth of pool4 for the later element wise addition
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1") # Add element wise the pool layer from the decoder

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(Sampled_image)
        # deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS]) #Set shape of the final deconvlution layer (shape of image depth number of class)
        W_t3 = utils.weight_variable([16, 16, 64, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([64], name="b_t3")
        # conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        Conv_t3a = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3,output_shape=[shape[0], shape[1], shape[2],64], stride=8)
        # FinalImage=tf.cast(conv_t3, tf.uint8,  name="ReconImage") %For 3d reconstrunction

        Relu1_2Shape = image_net["relu1_2"].get_shape()
        W_t3b = tf.Variable(tf.truncated_normal([3, 3, 64, 64], mean=0.0, stddev=0.01, dtype=tf.float32),"W_t3b")
        B_t3b = tf.Variable(tf.truncated_normal([64], mean=0.0, stddev=0.01, dtype=tf.float32), "B_t3b")
        Conv_t3b = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image_net["relu1_2"], W_t3b, [1, 1, 1, 1], padding="SAME"),B_t3b))

        Conv_t3fusion=tf.concat([Conv_t3a, Conv_t3b], 3)

        WFinal = tf.Variable(tf.truncated_normal([3, 3, 128, 3], mean=0.0, stddev=0.01, dtype=tf.float32),"WFinal")
        BFinal = tf.Variable(tf.truncated_normal([3], mean=0.0, stddev=0.01, dtype=tf.float32), "BFinal")
        ReconstructImage = tf.nn.bias_add(tf.nn.conv2d(Conv_t3fusion, WFinal, [1, 1, 1, 1], padding="SAME"), BFinal)

    return ReconstructImage

###########################################################################################################################################################
