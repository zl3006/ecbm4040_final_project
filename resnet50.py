
# Source Code: https://github.com/calmisential/TensorFlow2.0_ResNet
# Source Code: https://github.com/PeterWang512/CNNDetection
# Reference Paper: https://arxiv.org/abs/1512.03385

import tensorflow as tf
from tf.keras import layers


class BottleNeck(layers.Layer):

    def __init__(self, filter_num, stride = 1):

        super(BottleNeck, self).__init__()

        self.conv1 = layers.Conv2D(filters = filter_num, kernel_size = (1,1), strides = 1, padding = 'same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters = filter_num, kernel_size = (3,3), strides = stride, padding = 'same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filters = filter_num * 4, kernel_size = (1,1), strides = 1, padding = 'same')
        self.bn3 = layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()  
        self.downsample.add(layers.Conv2D(filters = filter_num * 4, kernel_size = (1,1), strides = stride))
        self.downsample.add(layers.BatchNormalization())


    def call(self, inputs, training = None, **kwargs):

        residual = self.downsample(inputs)     # should be same size as the network output to be added together.

        x = self.conv1(inputs)                 # 1x1 Convolution: reducing dimension. 
        x = self.bn1(x, training = training)   
        x = tf.nn.relu(x)                      

        x = self.conv2(x)                      # 3x3 Convolution: smaller input/output dimension.
        x = self.bn2(x, training = training)   
        x = tf.nn.relu(x)                     

        x = self.conv3(x)                      # 1x1 Convolution: increasing dimension.
        x = self.bn3(x, training = training)   

        x = layers.add([residual, x])          # should be same size as the network output to be added together.
        output = tf.nn.relu(x)                 

        return output


def bottleneck_layer(filter_num, blocks, stride = 1):

    resnet_block = tf.keras.Sequential()
    resnet_block.add(BottleNeck(filter_num, stride = stride))   # change dimension here!

    for _ in range(1, blocks):
        resnet_block.add(BottleNeck(filter_num, stride = 1))    # dimension stay the same.

    return resnet_block


class ResNet50(tf.keras.Model):

    def __init__(self, parameters):

        super(ResNetII, self).__init__()

        self.conv1 = layers.Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = "same")
        self.bn1 = layers.BatchNormalization()
        self.maxpool1 = layers.MaxPool2D(pool_size = (3,3), strides = 2, padding = "same")

        self.layer1 = bottleneck_layer(filter_num = 64, blocks = parameters[0])
        self.layer2 = bottleneck_layer(filter_num = 128, blocks = parameters[1], stride = 2)
        self.layer3 = bottleneck_layer(filter_num = 256, blocks = parameters[2], stride = 2)
        self.layer4 = bottleneck_layer(filter_num = 512, blocks = parameters[3], stride = 2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(units = 1, activation = tf.keras.activations.softmax)    # real or fake


    def call(self, inputs, training = None, mask= None):

        x = self.conv1(inputs)                    # input layers
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x, training = training)   # middle layers
        x = self.layer2(x, training = training)
        x = self.layer3(x, training = training)
        x = self.layer4(x, training = training)

        x = self.avgpool(x)                       # output layers 
        output = self.fc(x)

        return output


def resnet50():

    model = ResNet50(parameters = [3, 4, 6, 3])                                           
    return model
