#!/usr/bin/venv python

from symbol import parameters #chiedere a qualcuno spiegazioni per questo
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns

"""NOTA: per lo svolgimento del punto 2 dell'esercizio è necessario avere un
approccio di programmazione ad oggetti con l'utilizzo di classi e funzioni 
all'interno delle quali eseguire le operazionio fondamentali"""


"""NN MODEL FIT"""

#Building model
"""Naturalmente per poter utilizzare la funzione di iper-ottimizzazione devo definire quali
sono i paramtri che la funzione peò modificare durante l'ottimizzazione.
Questi parametri li inserisco in un vettore, scelgo come parametri da ottimizzare il 
learning rate e il numero di nodi in uno dei layers"""
def train(features, labels, parameters):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(units=parameters['layer_size'], activation='relu')) #NB: da controllare
    model.add(Dense(units=10, activation='relu')) #NB: da controllare
    model.summary()

    #Fitting model
    adam = Adam(lr=parameters["learning_rate"])
    model.compile(optimizer=adam, loss="categorical_crossentropy", #NB valutare lr
                    metrics=["accuracy"])
    model.fit(features, labels, epochs=5, verbose=1)
    #ho tolto il dataset di validation perchè creerò una funzione ad-hoc per questo

    return model

def test(model, features, labels):
    #Point 5: evaluate test accuracy
    scores = model.evaluate(features, labels)
    return scores

"""MAIN FUNCTION"""

def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # rescale data in interval [0,1]
    train_images = train_images / 255
    test_images = test_images / 255

    print('train:', train_images.shape)
    print('test:', test_images.shape)

    # build model
    """devo costruire una rete neurale anche all'interno del main perchè per
    fare il confronto voglio un modello statico, che quindi inizializzo nel main e 
    un modello dinamico che sarà quello che verrà ottimizzato"""
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    test_acc = test(model, test_images, test_labels)
    print(f'Test accuracy {test_acc}')

    """Costruisco il modello dinamico"""
    #implement hyperparameter tune with hyperopt
    test_setup = {
        'layer_size': 2,
        'learning_rate': 1.0
    }
    model = train(train_images, train_labels, test_setup)
    print('Accuracy on test set is:', test(model, test_images, test_labels)[1])

    #in pratica la funzione è esattamente come prima solo che la loss
    #non è deterministica ma deve essere calcolata dal modello della rete neurale 
    def hyper_func(params):
        model = train(train_images, train_labels, params)
        loss = -test_acc[1]
        
        return {
            'loss': loss,
            'status': STATUS_OK,
            'eval_time': time.time() }

    #define search space
    search_space = {
        'layer_size': hp.choice('layer_size', np.arange(10, 100, 20)),
        'learning_rate': hp.loguniform('learning_rate', -10, 0)
    }

    trials = Trials()
    best = fmin(hyper_func, search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    print(space_eval(search_space, best))


if __name__ == '__main__':
    main()