import csv
from scipy.sparse import csr_matrix
from numpy.linalg import norm
import numpy as np

def generate_data():
    #Matlab Portfolio Opt Data
    Sigma = np.loadtxt('dydata/sigma.txt')            
    m = np.loadtxt('dydata/m.txt')
    r = np.loadtxt('dydata/r.txt')
    gt = np.loadtxt('dydata/x_cvx_true.txt')
    eval = np.loadtxt('dydata/Sigma_eval.txt')

    return Sigma,r,m,gt,eval

class f1_copt:
    def __init__(self, H, Q):
        self.H = H
        self.Q = Q
    def __call__(self,x):
        return self.f_grad(x,return_gradient=False)
    def f_grad(self,x,return_gradient=True):
        Qx = self.Q @ x
        fval = 0.5 * x.T @ Qx
        if not return_gradient:
            return fval
        out = np.ravel(Qx)
        return fval, out

#Unit Simplex Projection
class f2_copt:
    def __init__(self, H):
        self.H  = H
    def __call__(self,x):
        return np.sum(x)
    def prox(self,x, step_size):
        step_size = np.real(step_size)
        idx_sort = np.argsort(x)[::-1]
        y = x[idx_sort]
        cumsum_list = (np.cumsum(y)-1)
        n = x.size
        temp        = np.arange(1,n+1)
        cumsum_list = cumsum_list/temp
        temp        = cumsum_list[np.sum(y>cumsum_list)-1]
        out         = np.maximum(x-temp,0)
        return np.ravel(out)

#Project Hyperplane
class f3_copt:
    def __init__(self, H,r,m):
        self.H = H
        self.r = r
        self.m = m
    def prox(self,x, step_size):
        nn = np.dot(self.m,self.H * self.m)
        out = x - np.maximum(np.dot(self.m/nn,x) + (self.r/nn),0)*self.m 
        return np.ravel(out)

