#!/usr/bin/venv python

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.datasets import load_iris

def BuildingModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(4,))),
    model.add(tf.keras.layers.Dense(128, activation='relu')),
    model.add(tf.keras.layers.Dense(128, activation='relu')),
    model.add(tf.keras.layers.Dense(128, activation='relu')),
    model.add(tf.keras.layers.Dense(64, activation='relu')),
    model.add(tf.keras.layers.Dense(64, activation='relu')),
    model.add(tf.keras.layers.Dense(64, activation='relu')),
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

def main():
    iris_data = load_iris()
    
    #load dataset in pandas frame e controllo
    iris_df = pd.DataFrame(iris_data.data, 
                           columns=iris_data.feature_names)
    iris_df["label"]=iris_data.target_names[iris_data.target]
    sns.pairplot(iris_df, hue='label')
    plt.show()


    """ONE-HOT ENCODING"""
    # label -> one-hot encoding
    label = pd.get_dummies(iris_df['label'], prefix='label')
    iris_df = pd.concat([iris_df, label], axis=1)
    iris_df.drop(['label'], axis=1, inplace=True)
    #check if worked
    column_headers = list(iris_df.columns.values)
    print("The Column Header :", column_headers)


    training_df = iris_df.sample(frac=0.8, random_state=1)
    test_df = iris_df.drop(training_df.index)

    """Nota: per poter effettivamente lavorare con i dati devo far sì
    che questi siano comprensibili al framework di tensorflow, quindi
    non possono rimanere in pandas.
    Dato che devo fare una classificazione devo dare alla rete delle informazioni
    in base alle quali imparare dei data-point (in sostanza un vettore 
    di dati) e le etichette di ciascun data point.
    La cosa principale è che tensorflow non legge i dataframe di pandas
    quindi devo trasformare le informazioni che mi interessano in vettori"""

    train_points = training_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    train_labels = training_df[['label_setosa', 'label_versicolor', 'label_virginica']]
    test_points = test_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    test_labels = test_df[['label_setosa', 'label_versicolor', 'label_virginica']]


    model = BuildingModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_points, train_labels, 
                        validation_split=0.4, epochs=200, verbose=1,  
                        batch_size = 32)

    
    #plot delle funzioni costo
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

    plt.savefig("DNNstandard.png")     
    fig.show()

    #plot score info
    scores = model.evaluate(test_points, test_labels)
    print('  Test loss:', scores[0])
    print('  Test accuracy:', scores[1])
    print(history.history)


    #use early stopping on vlidation loss with patience 10    
    model = BuildingModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_points, train_labels, 
                        validation_split=0.4, epochs=200, verbose=1,  
                        batch_size = 32, callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                            tf.keras.callbacks.TensorBoard(log_dir='./log')]
                        )


    #plot score info
    scores = model.evaluate(test_points, test_labels)
    print('  Test loss:', scores[0])
    print('  Test accuracy:', scores[1])
    print(history.history)

    #plot delle funzioni costo
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

    plt.savefig("DNNcallback.png")     
    fig.show()



if __name__ == '__main__':
    main()