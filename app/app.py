import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

import numpy as np

NOISE_DIM = 256

def Generator():
    gen_model = models.Sequential()

    gen_model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(NOISE_DIM,)))
    gen_model.add(layers.ReLU())
    gen_model.add(layers.Reshape((4, 4, 256)))

    gen_model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    gen_model.add(layers.BatchNormalization())
    gen_model.add(layers.ReLU())

    gen_model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    gen_model.add(layers.BatchNormalization())
    gen_model.add(layers.ReLU())
    
    gen_model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    gen_model.add(layers.BatchNormalization())
    gen_model.add(layers.ReLU())

    gen_model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return gen_model

generator = Generator()
generator.load_weights('app/waifu_generator.h5')

noise = tf.random.normal([1, NOISE_DIM])
predictions = generator(noise, training=False)
fig = plt.figure(figsize=(8, 8))

generated = (predictions[0, :, :, :])
plt.imshow(array_to_img(generated))
plt.axis('off')
plt.savefig(f'app/anime_pic.png')
plt.show()