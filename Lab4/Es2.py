#!/usr/bin/venv python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist as fash

"""DATA LOADING"""

#import datset and study data shape
(img_train, lab_train), (img_test, lab_test) = fash.load_data()

print('img_train shape:', img_train.shape)
print('lab_train shape:', lab_train.shape)
print('img_test shape:', img_test.shape)
print('lab_test shape:', lab_test.shape)

# dimensioni immagine di input
img_rows, img_cols = 28, 28 # number of pixels
num_classes = 10 #10 type of clothing items 

# cast floats to single precision
img_train = img_train.astype('float32')
img_test = img_test.astype('float32')

# rescale data in interval [0,1]
img_train /= 255
img_test /= 255

"""NN MODEL FIT"""

#Building model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.summary()

#Fitting model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

history = model.fit(img_train, lab_train, epochs=5, validation_data=(img_test, lab_test))

#Point 5: evaluate test accuracy
scores = model.evaluate(img_test, lab_test)
print('  Test loss:', scores[0])
print('  Test accuracy:', scores[1])
print(history.history)


fig, plts = plt.subplots(1,2)

plts[0].plot(history.history['loss'], label="training", color="blue")
plts[0].plot(history.history["val_loss"], label="validation", color="red")
plts[0].set_title("Loss functions")
plts[0].set_xlabel("epochs")
plts[0].set_ylabel("loss function")
plts[0].legend()
plts[0].grid()

plts[1].plot(history.history['accuracy'], label="training", color="blue")
plts[1].plot(history.history["val_accuracy"], label="validation", color="red")
plts[1].set_title("Accuracies")
plts[1].set_xlabel("epochs")
plts[1].set_ylabel("accuracy")
plts[1].legend()
plts[1].grid()

plt.savefig("output2.png")     
fig.show()

#Point 6: identify examples of bad classification