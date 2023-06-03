#!/usr/bin/venv python

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris

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

def CompileModel(model):
    model.compile(optimizer="Adam", loss="categorical_crossentropy", 
                    metrics=["loss"])
    model.summary()
    return model

def main():

    iris = load_iris()

    # load dataframe
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target_names[iris.target]

    print(df)
    sns.pairplot(df, hue='label')
    plt.show()

    # one-hot encoding
    # label -> one-hot encoding
    label = pd.get_dummies(df['label'], prefix='label')
    df = pd.concat([df, label], axis=1)
    # drop old label
    df.drop(['label'], axis=1, inplace=True)
    print(df)

    train_dataset = df.sample(frac=0.8, random_state=1)
    test_dataset = df.drop(train_dataset.index)

    # Creating X and y
    X_train = train_dataset[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

    # Convert DataFrame into np array
    y_train = train_dataset[['label_setosa', 'label_versicolor', 'label_virginica']]

    # build model
    model = BuildingModel()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # fit
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        validation_split=0.4,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            tf.keras.callbacks.TensorBoard(log_dir='./log'),
        ],
    )

    # plot metrics
    def plot_metric(history, metric):
        train_metrics = history.history[metric]
        val_metrics = history.history['val_'+metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation '+ metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_"+metric, 'val_'+metric])
        plt.show()

    plot_metric(history, 'loss')

if __name__ == '__main__':
    main()