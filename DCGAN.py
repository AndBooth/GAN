
import numpy as np
import os
import keras
import random
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Activation, ZeroPadding2D, BatchNormalization
from keras.models import Model, Sequential
from keras.activations import relu, softmax, tanh
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from utils import plotimgs

### A generator
# input: noise
# output: generated image

# dc generator

def generator(img_shape):

    inputs = Input(batch_shape = (None,100))

    X = Dense(7*7*128, activation = 'relu')(inputs)
    X = Reshape((7, 7, 128))(X)
    #X = Activation('relu')(X)
    X = UpSampling2D()(X)
    X = Conv2D(filters = 128, kernel_size = (3,3), strides = (1, 1), padding = "same")(X)
    X = BatchNormalization(momentum = 0.8)(X)
    X = Activation('relu')(X)
    X = UpSampling2D()(X)
    X = Conv2D(filters = 64, kernel_size = (3,3), strides = (1, 1), padding = "same")(X)
    X = BatchNormalization(momentum = 0.8)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = 1, kernel_size = (3,3), strides = (1, 1), padding = "same")(X)
    img = Activation('tanh')(X)

    return Model(inputs = inputs, outputs = img)

### A discriminator
# input: an image
# output: probability of fake or real

def discriminator(img_shape):

    inputs = Input(shape = (img_shape))

    X = Conv2D(32, kernel_size=3, strides=2, padding="same")(inputs)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(0.25)(X)
    X = Conv2D(64, kernel_size=3, strides=2, padding="same")(X)
    X = ZeroPadding2D(padding=((0,1),(0,1)))(X)
    X = BatchNormalization(momentum=0.8)(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(0.25)(X)
    X = Conv2D(128, kernel_size=3, strides=2, padding="same")(X)
    X = BatchNormalization(momentum=0.8)(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(0.25)(X)
    X = Conv2D(256, kernel_size=3, strides=1, padding="same")(X)
    X = BatchNormalization(momentum=0.8)(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(0.25)(X)
    X = Flatten()(X)
    pred = Dense(1, activation='sigmoid')(X)

    return Model(inputs = inputs, outputs = pred)

## A GAN
# inputs: random uniform noise from [0, 1]
# output: probability of fake or real

def gan(generator, discriminator):

    inputs = Input(batch_shape = (None,100))

    X = generator(inputs)
    pred = discriminator(X)

    return Model(inputs = inputs, outputs = pred)

shape = (28,28,1)

# Build generator and discriminator
gen = generator(shape)
disc = discriminator(shape)
gen.summary()
disc.summary()

# Optimisers
adam = optimizers.Adam(0.0002, 0.5)
# Pretrain discriminator
# noise_label = np.zeros(100)  # Fake labels for images
disc.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# Build GAN by passing generator and discriminator Models
gan = gan(gen, disc)
gan.summary()
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

## Training loop
steps = 5000
discK = 1  # Number of times to update discriminator weights per training step
batchsize = 32  # number of sampled fake and real images per step

# Import real images and normalise to [-1, 1]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
realimgs = x_train
realimgs = realimgs / 127.5 - 1.
realimgs = np.reshape(realimgs, (realimgs.shape[0], realimgs.shape[1], realimgs.shape[2], 1))

# Two methods to sample real images
# 1. Create permuted idxs of range of real images then use sequences from this in order
# permutatedidxs = np.random.permutation(realimgs.shape[0])
# 2. Randomly sample idxs per step  # CURRENTLY USING THIS ONE

for step in range(steps):

    # generate for this step
    stepnoise = np.random.uniform(-1, 1, size = (batchsize, 100))  # Noise
    steprealidxs = np.random.randint(0, realimgs.shape[0], batchsize)  # idxs to sample real images
    steprealimgs = realimgs[steprealidxs, :, :, :]

    stepfakeimgs = gen.predict(stepnoise)  # Generate fake images for this step
    #stepfakeimgs = np.reshape(stepfakeimgs, (stepfakeimgs.shape[0], stepfakeimgs.shape[1], stepfakeimgs.shape[2], 1))  

    # Make discriminator trainable to update its weights on fake and real images
    disc.trainable = True
    for layer in disc.layers:
        layer.trainable = True
    # train discriminator
    # two ways to try and train the discriminator
    # 1. Concatenate real and fake images and trian at same time 
    # 2. Train in separate batches - TRYING THIS ONE CURRENTLY
    # Can upate discriminator weights up to K times in loop
    for K in range(discK):
        stepdlossreal = disc.train_on_batch(steprealimgs, np.ones(batchsize))  # Label as 1s - i.e. true label
        stepdlossfake = disc.train_on_batch(stepfakeimgs, np.zeros(batchsize))  # Label as 0s - i.e. fake label - CAN TRY SWAPPING LABELS SOMETIMES

    # Set discriminator training to False to freeze its weights while training the GAN
    disc.trainable = False
    for layer in disc.layers:
        layer.trainable = False

    stepgloss = gan.train_on_batch(stepnoise, np.ones(batchsize))

    if step % 100 == 0:
        plotnoise = np.random.uniform(-1, 1, size = (16, 100))
        fakeplots = gen.predict(plotnoise)
        fakeplots = np.reshape(fakeplots, (fakeplots.shape[0], fakeplots.shape[1], fakeplots.shape[2]))
        plotimgs(fakeplots, step)
    #print(stepnoise)




