#!/usr/bin/venv python

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import PIL
from PIL import Image
import pathlib

"""CLASSIFICATION with DATA AUGMENTATION"""

def plot_image_samples(train_ds, name):

    #ho scelto una dimensione 4x8 perchè assumo una batch size di 32
    fig, axes = plt.subplots(4,8, figsize=(10, 10))
    class_names = train_ds.class_names
    for images, labels in train_ds.take(1): 
        #gli sto dicendo di prendere il primo batch di immagini
        axes = axes.flatten()
        for img, ax, lb in zip(images, axes, labels):
            #con la dicitura sopra gli sto dicendo in pratica
            #"fino a che non hai riempito tutti i posti"
            label_str = class_names[lb]
            ax.imshow(img.numpy().astype("uint8"))
            ax.set_title(label_str)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(name+".png")
    plt.show()

def directory_to_dataset(data_dir):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2,
        subset="training", seed=123, image_size=(224, 224), batch_size=32)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2,
        subset="validation", seed=123, image_size=(224, 224), batch_size=32)
    return train_ds, val_ds

def extract_img_lbs(dataset):
    images = []
    labels = []
    for img, lb in dataset.map(lambda x, y: (x, y)):
        images.append(img.numpy())
        labels.append(lb.numpy())

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    return images, labels

def BuildCNN(num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.Conv2D(16, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def plot_augmented_sample(train_dataset):
    data_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(224,224, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)])
    for images, labels in train_dataset.take(3):
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            aug_image = data_augmentation(images)
            plt.imshow(aug_image[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()



def Train(model, train_ds, val_ds):
    (train_images, train_labels) = extract_img_lbs(train_ds)
    (test_images, test_labels) = extract_img_lbs(val_ds)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="Adagrad", 
                  metrics=["acc"])
    model.summary()
    history=model.fit(train_images,train_labels, batch_size=32, epochs=8, 
                      verbose=1, validation_data=(test_images, test_labels))
    return history

def plot_history(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['loss'], label='train loss')
    ax1.plot(history.history['val_loss'], label='val loss')
    ax2.plot(history.history['acc'], label='train accuracy')
    ax2.plot(history.history['val_acc'], label='val accuracy')
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.savefig(name+".png")
    plt.show()

def model_withDA(num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.RandomRotation(0.1))
    model.add(tf.keras.layers.RandomZoom(0.1))
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.Conv2D(16, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model


def main():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    (train_ds, val_ds) = directory_to_dataset(data_dir)
    plot_image_samples(train_ds, "first_samples")
    num_classes = 5

    #optimize dataset performance cache and prefetch the original datasets
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = BuildCNN(num_classes)
    history = Train(model, train_ds, val_ds)
    plot_history(history, "Es1_history")

    plot_augmented_sample(train_ds)

    model = model_withDA(num_classes)
    history = Train(model, train_ds, val_ds)
    plot_history(history, "Es1_history_augmented")


if __name__ == '__main__':
    main()
