"""
Estimating a sparse and low rank matrix
=======================================

"""
import copt.loss
import copt.penalty

import numpy as np
from scipy.sparse import linalg as splinalg
import matplotlib.pyplot as plt
import copt as cp

def generate_data():
    # .. Generate synthetic data ..
    np.random.seed(1)
    
    sigma_2 = 0.6
    N = 100
    d = 20
    blocks = np.array([2 * d / 10, 1 * d / 10, 1 * d / 10, 3 * d / 10, 3 * d / 10]).astype(
        np.int
    )
    epsilon = 10 ** (-15)
    
    mu = np.zeros(d)
    Sigma = np.zeros((d, d))
    blck = 0
    for k in range(len(blocks)):
        v = 2 * np.random.rand(int(blocks[k]), 1)
        v = v * (abs(v) > 0.9)
        Sigma[blck : blck + blocks[k], blck : blck + blocks[k]] = np.dot(v, v.T)
        blck = blck + blocks[k]
    X = np.random.multivariate_normal(
        mu, Sigma + epsilon * np.eye(d), N
    ) + sigma_2 * np.random.randn(N, d)
    Sigma_hat = np.cov(X.T)
    
    threshold = 1e-5
    Sigma[np.abs(Sigma) < threshold] = 0
    Sigma[np.abs(Sigma) >= threshold] = 1
    
    # .. generate some data ..
    
    max_iter = 5000
    
    n_features = np.multiply(*Sigma.shape)
    n_samples = n_features
    print("#features", n_features)
    A = np.random.randn(n_samples, n_features)
    
    sigma = 1.0
    b = A.dot(Sigma.ravel()) + sigma * np.random.randn(n_samples)
    
    # .. compute the step-size ..
    s = splinalg.svds(A, k=1, return_singular_vectors=False, tol=1e-3, maxiter=500)[0]
    f = copt.loss.HuberLoss(A, b)
    step_size = 1.0 / f.lipschitz
    return A,b,f,Sigma,step_size, n_features;

def loss(x, f_sp, G1, G2):
    return f_sp(x) + G1(x) + G2(x)
