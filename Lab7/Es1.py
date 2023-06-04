#!/usr/bin/venv python

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import PIL
from PIL import Image


#Allocate a sequential model with a normalization layer
#First try with a DNN
def BuildDNN(num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu")) #capire come conviene costruirlo
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax")) #final softmax layer
    return model


#Second try with a CNN  
def BuildCNN(num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def Train(model, train_images, train_labels, test_images, test_labels):
    model.compile(loss="sparse_categorical_crossentropy", optimizer="Adagrad", 
                  metrics=["acc"])
    model.summary()
    history=model.fit(train_images,train_labels, batch_size=32, epochs=10, 
                      verbose=1, validation_data=(test_images, test_labels))
    return history 

#function that plots image samples
def plot_image_samples(images, labels, name):

    num_samples=25
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_title(f"Class: {labels[i][0]}")
        if i == num_samples-1:
            break
    plt.savefig(name+".png")
    plt.show()

#function to get the number of pixels in an inmage
def PixelsRange(image):
    return [np.min(image), np.max(image)]
     
#function to plot loss and accuracy of two architetcture given history
def CompareResult(history1, history2):

    fig, plts = plt.subplots(2,2)
    plts[0][0].plot(history1.history['acc'], label="training", color="blue")
    plts[0][0].plot(history2.history["acc"], label="validation", color="red")
    plts[0][0].set_title("Accuracy functions")
    plts[0][0].set_xlabel("epochs")
    plts[0][0].set_ylabel("accuracy")
    plts[0][0].legend()
    plts[0][0].grid()

    plts[0][1].plot(history1.history['val_acc'], label="training", color="blue")
    plts[0][1].plot(history2.history["val_acc"], label="validation", color="red")
    plts[0][1].set_title("Validation accuracy")
    plts[0][1].set_xlabel("epochs")
    plts[0][1].set_ylabel("accuracy")
    plts[0][1].legend()
    plts[0][1].grid()

    plts[1][0].plot(history1.history['loss'], label="training", color="blue")
    plts[1][0].plot(history2.history["loss"], label="validation", color="red")
    plts[1][0].set_title("Loss functions")
    plts[1][0].set_xlabel("epochs")
    plts[1][0].set_ylabel("loss function")
    plts[1][0].legend()
    plts[1][0].grid()

    plts[1][1].plot(history1.history['val_loss'], label="training", color="blue")
    plts[1][1].plot(history2.history["val_loss"], label="validation", color="red")
    plts[1][1].set_title("Loss functions")
    plts[1][1].set_xlabel("epochs")
    plts[1][1].set_ylabel("loss function")
    plts[1][1].legend()
    plts[1][1].grid()

    plt.savefig("Results.png")
    plt.show()    


def main():
    num_class = 10
    img_rows = 32
    img_cols = 32
    data = tf.keras.datasets.cifar10.load_data
    print(type(data))

    #create data that can be given as input to my CNN
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    plot_image_samples(train_images, train_labels, "first_samples")

    #Building&Training
    model = BuildDNN(num_class)
    history1 =  Train(model, train_images, train_labels, test_images, test_labels)

    #Building&Training CNN
    modelCNN = BuildCNN(num_class)
    history2 = Train(modelCNN, train_images, train_labels, test_images, test_labels)

    CompareResult(history1, history2)

if __name__ == '__main__':
    main()