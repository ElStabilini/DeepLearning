#!/usr/bin/venv python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras
import hyperopt
from hyperopt import hp
import hyperopt.pyll.stochastic as hpst
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
import time
import seaborn as sns

#cosmesi dei plot
sns.set(rc = {'figure.figsize':(12,7),  "legend.fontsize": 8, "legend.title_fontsize": 10})

"""DEFINE OBJECTIVE FUNCTION"""
def objective(x):
    return {
        'loss': 0.05*(x**6 - 2*x**5 - 28*x**4 + 28*x**3 + 12*x**2 -26*x + 100),
        'status': STATUS_OK,
        'eval_time': time.time() }

x = np.linspace(-5,6,200)
y = objective(x)["loss"]

"""PLOT FUNCTION"""
plt.plot(x, y, color = "turquoise", label="objective function")
plt.title("Data")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-5,6)
plt.legend()
plt.savefig("output1a.png")
plt.show()

"""INITIALIZE HYPEROPT"""
#defining hyperopt space
space=hp.uniform('x', -5, 6)

#build a sample histogram
samples = [hpst.sample(space) for i in range(1000)]

plot=sns.histplot(samples, bins=30, color="mediumaquamarine")
plot.set_xlabel('number')
plot.set_ylabel('occurency')
plot.set_title("Esplorazione delllo spazio con hyperopt")
fig = plot.get_figure()
fig.savefig('output1b.png')


"""OBJECTIVE FUNCTION MINIMIZATION"""
algorithm = tpe.suggest
trials = Trials()
best = fmin(objective, space=space, algo=algorithm, max_evals=2000, trials=trials)
print(best)

#scatter plot
iterations = trials.idxs_vals[0]['x']
x_values = trials.idxs_vals[1]['x']
plt.scatter(iterations, x_values, color="darkmagenta")
plt.title("hyperopt space exploration")
plt.xlabel("iteration")
plt.ylabel("x-value")
plt.ylim(-8,8)
plt.savefig("std_scatter.png")
plt.show()

#histplot
plot=sns.histplot(x_values, bins=30, color="mediumaquamarine")
plot.set_xlabel('number')
plot.set_ylabel('occurency')
plot.set_title("Esplorazione delllo spazio con hyperopt")
fig = plot.get_figure()
fig.savefig('std_opt_hist.png')



"""STESSO ESERCIZIO CON RAND"""
algorithm = rand.suggest
trials = Trials()
best = fmin(objective, space=space, algo=algorithm, max_evals=2000, trials=trials)
print(best)

#scatter plot
iterations = trials.idxs_vals[0]['x']
x_values = trials.idxs_vals[1]['x']
plt.scatter(iterations, x_values, color="darkmagenta")
plt.title("hyperopt space exploration")
plt.xlabel("iteration")
plt.ylabel("x-value")
plt.ylim(-8,8)
plt.savefig("rnd_scatter.png")
plt.show()

#histplot
plot=sns.histplot(x_values, bins=30, color="mediumaquamarine")
plot.set_xlabel('number')
plot.set_ylabel('occurency')
plot.set_title("Esplorazione delllo spazio con hyperopt")
fig = plot.get_figure()
fig.savefig('rnd_opt_hist.png')