import time
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials

def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK, 'eval_time': time.time()}

algorithm = tpe.suggest # or rand.suggest
trials = Trials() # objecting collecting sequential trials
best = fmin(objective, space=hp.uniform('x', -10, 10),
algo=algorithm, max_evals=100,
trials=trials)
print(best) # returns dict with {'x': value}
iterations = trials.idxs_vals[0]['x']
x_values = trials.idxs_vals[1]['x']
