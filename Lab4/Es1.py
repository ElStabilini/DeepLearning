#!/usr/bin/venv python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras


""" DATA LOADING """

x_tr, y_tr, x_val, y_val = np.loadtxt("./data.dat", usecols=(0,1,2,3), unpack="True")
#NB: anche per il futuro ricordati di caricare i dati su array di numpy altrimenti da problemi

"""
    BASELINE LINEAR FIT
    fit lineare significa che le funzioni di attivazione
    della rette sono tutte lineari
"""

def LinearFit():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                            loss=tf.keras.losses.mean_squared_error)

    return model


"""
    NN model fit
"""

def NeuralNetwork():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    model = tf.keras.Sequential()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.mean_squared_error)
   
    return model


#Inizio stampa e salvataggio output

plt.scatter(x_tr, y_tr, color = "turquoise", label="Training Data")
plt.scatter(x_val, y_val, color='blue', label="Validation Data")
plt.title("Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.savefig("output1a.png")
plt.show()

fig, plts = plt.subplots(1,2)

#PLOT first part of the exercise
model1 = LinearFit()
history1 =  model1.fit(x_tr, y_tr, epochs=500, batch_size=len(x_tr),
            validation_data=(x_val, y_val))

plts[0].plot(history1.history['loss'], label="training", color="turquoise")
plts[0].plot(history1.history["val_loss"], label="validation", color="blue")
plts[0].set_title("Loss function")
plts[0].set_xlabel("epochs")
plts[0].set_ylabel("loss function")
plts[0].legend()

#PLOT the second part of the excercise

model2 = NeuralNetwork()
history2 =  model1.fit(x_tr, y_tr, epochs=500, batch_size=len(x_tr),
            validation_data=(x_val, y_val))

plts[1].plot(history2.history['loss'], label="training", color="turquoise")
plts[1].plot(history2.history["val_loss"], label="validation", color="blue")
plts[1].set_title("Loss function")
plts[1].set_xlabel("epochs")
plts[1].set_ylabel("loss function")
plts[1].legend()

plt.savefig("output1b.png")     
fig.show()