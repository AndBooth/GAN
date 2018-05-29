
import numpy as np
import os
import keras
import random
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model, Sequential
from keras.activations import relu, softmax
from keras.datasets import mnist

### A generator
# input: noise
# output: generated image

def generator(img_shape):

    inputs = Input(batch_shape = (None,100))

    X = Dense(64, activation = 'relu')(inputs)
    X = Dense(128, activation = 'relu')(X)
    X = Dense(np.prod(img_shape), activation = 'tanh')(X)
    img = Reshape(img_shape)(X)

    return Model(inputs = inputs, outputs = img)

### A discriminator
# input: an image
# output: probability of fake or real

def discriminator(img_shape):

    inputs = Input(shape = (img_shape))

    X = Flatten()(inputs)
    X = Dense(np.prod(img_shape), activation = 'relu')(X)
    X = Dense(128, activation = 'relu')(X)
    X = Dense(64, activation = 'relu')(X)
    pred = Dense(1, activation = 'sigmoid')(X)

    return Model(inputs = inputs, outputs = pred)

## A GAN
# inputs: random uniform noise from [0, 1]
# output: probability of fake or real

def gan(generator, discriminator):

    inputs = Input(batch_shape = (None,100))

    X = generator(inputs)
    pred = discriminator(X)

    return Model(inputs = inputs, outputs = pred)

shape = (28,28)
np.random.seed(123)
noise = np.random.uniform(0, 1, size = [100,100])  # Random uniform noise from [0, 1]

# Build generator and discriminator
gen = generator(shape)
disc = discriminator(shape)
gen.summary()
disc.summary()

# Optimisers
adam = optimizers.Adam(lr = 0.001)

# Pretrain discriminator
# imgs = gen.predict(noise)  # Create some fake images
# noise_label = np.zeros(100)  # Fake labels for images
disc.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# disc.fit(imgs, noise_label)

# preds = disc.predict(imgs)
# print(preds)

# Build GAN by passing generator and discriminator Models
gan = gan(gen, disc)
gan.summary()
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

gan_out = gan.predict(noise)

print(gan_out.shape)
#print(disc_out)


## Training loop
steps = 10000
batchsize = 128  # number of sampled fake and real images per step

# Import real images
(x_train, y_train), (x_test, y_test) = mnist.load_data()
realimgs = np.concatenate((x_train, x_test), axis = 0)

# Two methods to sample real images
# 1. Create permuted idxs of range of real images then use sequences from this in order
# permutatedidxs = np.random.permutation(realimgs.shape[0])
# 2. Randomly sample idxs per step  # CURRENTLY USING THIS ONE

for step in range(steps):

    # generate for this step
    stepnoise = np.random.uniform(0, 1, size = (batchsize, 100))  # Noise
    steprealidxs = np.random.randint(0, realimgs.shape[0], batchsize)  # idxs to sample real images
    steprealimgs = realimgs[steprealidxs, :, :]

    stepfakeimgs = gen.predict(stepnoise)  # Generate fake images for this step

    # Make discriminator trainable to update its weights on fake and real images
    disc.trainable = True
    for layer in disc.layers:
        layer.trainable = True
    # train discriminator
    # two ways to try and train the discriminator
    # 1. Concatenate real and fake images and trian at same time 
    # 2. Train in separate batches - TRYING THIS ONE CURRENTLY
    stepdlossreal = disc.train_on_batch(steprealimgs, np.ones(batchsize))  # Label as 1s - i.e. true label
    stepdlossfake = disc.train_on_batch(stepfakeimgs, np.zeros(batchsize))  # Label as 0s - i.e. fake label - CAN TRY SWAPPING LABELS SOMETIMES

    # Set discriminator training to False to freeze its weights while training the GAN
    disc.trainable = False
    for layer in disc.layers:
        layer.trainable = False

    stepgloss = gan.train_on_batch(stepnoise, np.ones(batchsize))

    if step % 200 == 0:
        plotnoise = np.random.uniform(0, 1, size = (16, 100))
        fakeplots = gen.predict(plotnoise)

        plt.figure(figsize=(10,10))  # todo: write plotting as a function
        for i in range(fakeplots.shape[0]):
            plt.subplot(4, 4, i+1)
            image = fakeplots[i, :, :]
            #image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
            
        plt.savefig('plot_' + str(step) + ".png" )
        plt.close('all')

    #print(stepnoise)




