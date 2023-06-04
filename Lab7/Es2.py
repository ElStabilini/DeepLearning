#!/usr/bin/venv python

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import PIL
from PIL import Image
from matplotlib.patches import Rectangle

"""Construct a custom Keras model using functional API with
1. a feature extractor:
    *   a CNN
    *   a flatten layer
    *   a dense layer
2.  a classifier (10 classes)
3.  a bounding box regressor (4 coordinates)
"""
#function to load data
def load_data():
    data_dir = "./data"
    valid_images = np.load(f"{data_dir}/validation_images.npy")
    valid_labels = np.load(f"{data_dir}/validation_labels.npy")
    valid_boxes = np.load(f"{data_dir}/validation_boxes.npy")
    train_images = np.load(f"{data_dir}/training_images.npy")
    train_labels = np.load(f"{data_dir}/training_labels.npy")
    train_boxes = np.load(f"{data_dir}/training_boxes.npy")
    
    return valid_images, train_images, valid_labels, train_labels, train_boxes, valid_boxes

#function that plots image samples
def plot_image_samples(images, boxes, pboxes=None, plot_predictions=False):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap='binary')
        center = (boxes[i, 0]*75, boxes[i, 1]*75)
        plt.gca().add_patch(Rectangle(center,
                                      (boxes[i, 2]-boxes[i, 0])*75,
                                      (boxes[i, 3]-boxes[i, 1])*75,
                                      edgecolor='red',
                                      facecolor='none',
                                      lw=4))
        if plot_predictions:
            center = (pboxes[i, 0]*75, pboxes[i, 1]*75)
            plt.gca().add_patch(Rectangle(center,
                                        (pboxes[i, 2]-pboxes[i, 0])*75,
                                        (pboxes[i, 3]-pboxes[i, 1])*75,
                                        edgecolor='blue',
                                        facecolor='none',
                                        lw=4))
    plt.savefig("second_samples")
    plt.show()

def plot_history(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['classifier_loss'], label='classifier loss')
    ax1.plot(history.history['val_classifier_loss'], label='val classifier loss')
    ax2.plot(history.history['bounding_box_loss'], label='bounding box loss')
    ax2.plot(history.history['val_bounding_box_loss'], label='val bounding box loss')
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.savefig(name+".png")
    plt.show()

"""L'idea in questo caso è quella di creare un modello unico che poi
viene utilizzato per ottenere due output diversi uno per la classificazione
e uno per la localizzazione.

Il modello unico sarà quello che ricava le caratteristiche principali
delle immagini qunidi il feature estractor.

Questo modello verrà poi passato come input ad una seconda rete
più piccola (in pratica alleno dei pezzi di architettura separatamente)"""

def create_model():
    inputs = tf.keras.layers.Input(shape=(75, 75, 1,))
    dense_output = feature_extractor(inputs)
    classification_output = classifier(dense_output)
    regression_output = regressor(dense_output)
    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, regression_output])
    return model

def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x

def classifier(inputs):
    return tf.keras.layers.Dense(10, activation='softmax', name='classifier')(inputs)

def regressor(inputs):
    return tf.keras.layers.Dense(4, name='bounding_box')(inputs)

def Train(model, train_images, train_labels, train_boxes, valid_images, valid_labels, valid_boxes):
    model.compile(optimizer='adam', 
                    loss={
                      'classifier': 'categorical_crossentropy',
                      'bounding_box': 'mse'
                    },
                    metrics={
                      'classifier': 'acc',
                      'bounding_box': 'mse'
                  })
    model.summary()
    history = model.fit(train_images, (train_labels, train_boxes),
                        validation_data=(valid_images, (valid_labels, valid_boxes)),
                        epochs=10)
    return history


def intersection_over_union(true_box, pred_box):
    #estraggo le coordinate dei boxes
    xmin_pred, ymin_pred, xmax_pred, ymax_pred =  np.split(pred_box, 4, axis = 1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis = 1)

    smoothing_factor = 1e-12

    #controllo l'overlap
    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0)  * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou 

def iou_eval(iou, soglia=0.6):
    good = 0
    bad = 0
    for i in iou:
        if i >= soglia:
            good +=1
            continue
        bad += 1
    return good, bad

def main():
    valid_images, train_images, valid_labels, train_labels, train_boxes, valid_boxes = load_data()

    plot_image_samples(train_images, train_labels, train_boxes, "first_samples_box")

    model = create_model()
    history = Train(model, train_images, train_labels, train_boxes, valid_images, valid_labels, valid_boxes)
    plot_history(history, "Es2_Results")

    predictions = model.predict(valid_images)
    plot_image_samples(valid_images, valid_boxes, predictions[1], plot_predictions=True)

    iou = intersection_over_union(valid_boxes, predictions[1])
    good, bad = iou_eval(iou)
    print('Number of good bounding box prediction: ', good)
    print('Number of bad bounding box prediction: ', bad)    

if __name__ == "__main__":
    main()