"""
Group lasso with overlap
========================

Comparison of solvers for a least squares with
overlapping group lasso regularization.

References
----------
This example is modeled after the experiments in `Adaptive Three Operator Splitting <https://arxiv.org/pdf/1804.02339.pdf>`_, Appendix E.3.
"""
import copt as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

import copt.loss
import copt.penalty
from libsvm.svmutil import *

def generate_data():
    np.random.seed(0)
    
    n_samples, n_features = 100, 1002
    
    # .. generate some data ..
    # .. the first set of blocks is
    groups = [np.arange(8 * i, 8 * i + 10) for i in range(125)]
    ground_truth = np.zeros(n_features)
    g = np.random.randint(0, len(groups), 10)
    for i in g:
        ground_truth[groups[i]] = np.random.randn()
    
    A = np.random.randn(n_samples, n_features)
    p = 0.95  # create a matrix with correlations between features
    for i in range(1, n_features):
        A[:, i] = p * A[:, i] + (1 - p) * A[:, i-1]
    A[:, 0] /= np.sqrt(1 - p ** 2)
    A = preprocessing.StandardScaler().fit_transform(A)
    b = A.dot(ground_truth) + np.random.randn(n_samples)
    
    # make labels in {0, 1}
    b = np.sign(b)
    b = (b + 1) // 2
    
    
    # .. compute the step-size ..
    max_iter = 5000
    f = copt.loss.LogLoss(A, b)
    step_size = 1. / f.lipschitz

    return A,b,f,groups,ground_truth,n_features,step_size


def loss(x,f,G1,G2):
    return f(x) + G1(x)+ G2(x)
