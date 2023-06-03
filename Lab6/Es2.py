#!/usr/bin/venv python

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.datasets import load_iris

def BuildFit(train_data, train_label):
    
    #build model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(30, activation='relu', input_shape=(10,1))),
    model.add(tf.keras.layers.Dense(1))

    #compile model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    model.summary()

    model.fit(train_data, train_label, epochs=25, verbose=1,  batch_size=32)
    return model

def TestModel(model, test_data, test_label):
    scores = model.evaluate(test_data, test_label)
    return scores

def main():
    
    #import data
    data_tr = np.load("./training_data.npy")
    lab_tr = np.load("./training_label.npy")
    data_test = np.load("./test_data.npy")
    lab_test = np.load("./test_label.npy")

    print(data_tr[0])
    print(lab_tr[0])
    print(data_tr[0].shape)
    print(lab_tr[0].shape)
    print(len(data_tr))
    print(len(lab_tr))
    print(len(data_test))
    print(len(lab_test))
    
    MyModel = BuildFit(data_tr,lab_tr)
    #Test = TestModel(MyModel, data_test, lab_test)
    lab_pred = MyModel.predict(data_test)

    """PLOTS"""
    # Show results
    plt.figure(figsize=(10,8))

    plt.subplot(3, 1, 1)
    plt.plot(lab_tr, label='Train data')
    plt.plot(range(len(lab_tr), len(lab_tr)+len(lab_test)), lab_test, 'k', label='Test data')
    plt.legend(frameon=False)
    plt.ylabel("Temperature")
    plt.xlabel("Day")
    plt.title("All data")

    plt.subplot(3, 2, 3)
    plt.plot(lab_test, color='k', label = 'True value')
    plt.plot(lab_pred, color='red', label = 'Predicted')
    plt.legend(frameon=False)
    plt.ylabel("Temperature")
    plt.xlabel("Day")
    plt.title("Predicted data (full test set)")

    plt.subplot(3, 2, 4)
    plt.plot(lab_test[0:100], color='k', label = 'True value')
    plt.plot(lab_pred[0:100], color = 'red', label='Predicted')
    plt.legend(frameon=False)
    plt.ylabel("Temperature")
    plt.xlabel("Day")
    plt.title("Predicted data (first 100 days)")

    plt.subplot(3, 2, 5)
    plt.plot(lab_test-lab_pred, color='k')
    plt.ylabel("Residual")
    plt.xlabel("Day")
    plt.title("Residual plot")

    plt.subplot(3, 2, 6)
    plt.scatter(lab_pred, lab_test, s=2, color='black')
    plt.ylabel("Y true")
    plt.xlabel("Y predicted")
    plt.title("Scatter plot")

    mse = np.mean(np.square(lab_test - lab_pred))
    print(f"MSE = {mse}")

    plt.subplots_adjust(hspace = 0.5, wspace=0.3)
    plt.savefig("AllData.png")     
    plt.show()

if __name__ == '__main__':
    main()