
import numpy as np
import os
import keras
from keras import optimizers
from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model, Sequential
from keras.activations import relu, softmax

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

    # Set discriminator training to False to freeze its weights
    discriminator.trainable = False
    for layer in discriminator.layers:
        layer.trainable = False

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
imgs = gen.predict(noise)  # Create some fake images
noise_label = np.zeros(100)  # Fake labels for images
disc.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
disc.fit(imgs, noise_label)

preds = disc.predict(imgs)
print(preds)

# Build GAN by passing generator and discriminator Models
gan = gan(gen, disc)
gan.summary()

gan_out = gan.predict(noise)

print(gan_out.shape)
#print(disc_out)
