import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Model, regularizers
import tensorflow_addons as tfa
import skimage.transform as st

MEAN = 7433.6436
STD = 2353.0723
LR_SIZE = 34
LR_SIZE_2 = 66
LR_SIZE_3 = 130

def normalize(x):
    return (x-MEAN)/STD

def denormalize(x):
    return x * STD + MEAN 

def upsample(x, **kwargs):
    return layers.UpSampling2D(trainable=False, size=(2, 2), interpolation='nearest', **kwargs)(x)

def conv3d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(layers.Conv3D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def dense32_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    nn = tf.keras.Sequential()
    nn.add(layers.Flatten())
    nn.add(tfa.layers.WeightNormalization(layers.Dense(units=32*32,activation=activation, **kwargs), data_init=False))
    return nn


def dense64_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    nn = tf.keras.Sequential()
    nn.add(layers.Flatten())
    nn.add(tfa.layers.WeightNormalization(layers.Dense(units=64*64,activation=activation, **kwargs), data_init=False))
    return nn

def wdsr_3d(scale, num_filters, num_res_blocks, res_block_expansion,channels):
    img_inputs = Input(shape=(LR_SIZE, LR_SIZE,channels, 1))

    x = layers.Lambda(normalize)(img_inputs)

    m = conv3d_weightnorm(1, (3,3,channels),padding='valid',activation='relu')(x)
    x=m

    # Laplacian pyramid

    #First layer
    #Extraction
    m = dense32_weightnorm(num_filters, (5,5,1),padding='same',activation='relu')(m)
    m = layers.Reshape((32,32,1,1))(m)
    #Upsampling
    m = conv3d_weightnorm(4, (3,3,1),padding='valid')(m)
    m = layers.Lambda(lambda x: tf.pad(x,[[0,0],[1,1],[1,1],[0,0],[0,0]],mode='REFLECT'))(m)
    m = layers.Reshape((LR_SIZE-2,LR_SIZE-2,4))(m)
    m = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(m)
    m = layers.Reshape((LR_SIZE_2-2,LR_SIZE_2-2,1,1))(m)

    x=layers.Reshape((LR_SIZE-2,LR_SIZE-2,1))(x)
    x=upsample(x)
    x=layers.Reshape((LR_SIZE_2-2,LR_SIZE_2-2,1,1))(x)
    x=x+m;

    #Second Layer
    #Extraction
    m = dense64_weightnorm(num_filters, (5,5,1),padding='same',activation='relu')(m)
    m = layers.Reshape((64,64,1,1))(m)
    #Upsampling
    m = conv3d_weightnorm(4, (3,3,1),padding='valid')(m)
    m = layers.Lambda(lambda x: tf.pad(x,[[0,0],[1,1],[1,1],[0,0],[0,0]],mode='REFLECT'))(m)
    m = layers.Reshape((LR_SIZE_2-2,LR_SIZE_2-2,4))(m)
    m = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(m)
    x=layers.Reshape((LR_SIZE_2-2,LR_SIZE_2-2,1))(x)
    x=upsample(x)
    m=x+m;

    m = tf.image.resize(m,[96,96])
    
    outputs = m
    outputs = layers.Lambda(denormalize)(outputs)

    return Model([img_inputs], outputs, name="manhattan")
