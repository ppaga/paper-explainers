from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from scipy.special import logit, expit
from numpy.random import randn, rand
import numpy as np
from collections import namedtuple

import matplotlib.pyplot as plt

def LR_vars(**kwargs):
    X = np.random.randn(kwargs['N'],kwargs['theta'].shape[0])
    Y = np.matmul(X,kwargs['theta'])/np.sqrt(kwargs['theta'].shape[0]) + kwargs['sigma']*randn(kwargs['N'])
    return X,Y

LR_key = namedtuple('LR_key',['N','alpha','sigma'])
def run_LR_simulation(N, sigma, alphas):
  experiments = {}
  for alpha in alphas:
    d = int(N*alpha)
    theta = randn(d)

    key = LR_key(N,alpha,sigma)

    X_train,Y_train = LR_vars(N=N, theta = theta, sigma = sigma)
    X_test,Y_test = LR_vars(N=N, theta = theta, sigma = sigma)
    
    model = LinearRegression().fit(X_train,Y_train)
    theta_tilde, Y_train_tilde, Y_test_tilde = model.coef_, model.predict(X_train), model.predict(X_test)
    experiments[key] = theta_tilde, theta, Y_train_tilde, Y_train, Y_test_tilde, Y_test
  return experiments
  

def plot_results(experiments):
  fig, axes = plt.subplots(1,3,figsize = (18,6))
  for key,val in experiments.items():
    theta_tilde, theta, Y_train_tilde, Y_train, Y_test_tilde, Y_test = val
    axes[0].scatter(Y_train, Y_train_tilde, s = 5)
    axes[0].scatter(Y_test, Y_test_tilde, s = 5)
    axes[1].scatter(theta, theta_tilde, s = 5)
  plt.legend([np.round(key.alpha, decimals=3) for key in experiments.keys()])
