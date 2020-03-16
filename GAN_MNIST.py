import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#plt.imshow(train_images[0])
#plt.show()

print(train_images[0])
print(train_images.shape)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
print(train_images.shape)

train_images = (train_images-127.5)/127.5

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 100
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Discriminator Model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(7,(3,3), padding="same", input_shape=(28,28,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    return model

model_discriminator = make_discriminator_model()

model_discriminator(tf.convert_to_tensor(np.random.rand(1, 28, 28, 1).astype("float32")))
print(model_discriminator)

