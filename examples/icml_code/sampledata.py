from scipy.sparse import csr_matrix
from numpy.linalg import norm
import numpy as np
import scipy as sp

class f_copt:
    #Anisotropic Forward Operator Least Square
    # 0.5\|Fx-b\|_2^2
    # F =[C_11, C_12; 0, C_22]
    # \|C_11\| = k1
    # \|C_22\| = k2
    # C_xx = n x n
    def __init__(self,n,k1,k2):
        self.n = n
        self.k1 = k1
        self.k2 = k2
        np.random.seed(1111)
        q1 = sp.stats.ortho_group.rvs(n)
#        A = np.random.rand(n,n)
        np.random.seed(1112)
        b = np.random.rand(2*n)
        np.random.seed(1113)
        l3 = np.random.rand(n)
        np.random.shuffle(l3)


        np.random.seed(1114)
        q2 = sp.stats.ortho_group.rvs(n)
        np.random.seed(1115)
        q3 = sp.stats.ortho_group.rvs(n)
        if k1 == 1:
            l1 = np.random.rand(n)
        else:
            l1 = np.linspace(1,k1,n)
        np.random.shuffle(l1)
        l2 = np.linspace(1,k2,n)
        np.random.shuffle(l2)
        C11 = q1 @ (np.diag(l1) @ q1.T)
        C12 = q2 @ (np.diag(l3) @ q2.T)
        C22 = q3 @ (np.diag(l2) @ q3.T)
        zr  = np.zeros((n,n))
        temp1 = np.concatenate((C11,C12),axis=1)
        temp2 = np.concatenate((zr,C22),axis=1)
        out = np.concatenate((temp1,temp2),axis=0)
        self.F = out
        self.b = b
        self.FTb = out.T @ b
        FTF = out.T @ out
        w1,v1, = np.linalg.eig(FTF)
        self.mineig = np.min(w1)
        self.maxeig = np.max(w1)
    def __call__(self,x):
        return self.f_grad(x,return_gradient=False)
    def Lip(self):
        return self.maxeig
    def f_grad(self,x,return_gradient=True):
        Fx = self.F @ x
        fval = 0.5*norm(Fx-self.b)**2
        if not return_gradient:
            return fval
        FTFx = self.F.T @ Fx
        out = np.ravel(FTFx - self.FTb)
        return fval, out
    def data(self):
        return self.F, self.b

class g_copt:
    def __init__(self, c):
        self.c = c
    def __call__(self,x):
        return self.f_grad(x,return_gradient=False)
    def prox(self,x, step_size):
        step_size = np.real(step_size)
        out = x - step_size*self.c
        return np.ravel(out)
    def f_grad(self,x,return_gradient=True):
        fval = np.dot(self.c,x)
        if not return_gradient:
            return fval
        return fval, self.c

class h_copt:
    def __init__(self, mu, H):
        self.mu = mu
        self.H  = H
    def __call__(self,x):
        return self.f_grad(x,return_gradient=False)
    def prox(self,x, step_size):
        step_size = np.real(step_size)
        out = (x + np.sqrt(x**2 + 4*self.mu*step_size*self.H))/2
        return np.ravel(out)
    def f_grad(self,x,return_gradient=True):
        fval = -self.mu*np.sum(np.log(x))
        if not return_gradient:
            return fval
        out = -self.mu/x
        out = np.ravel(out)
        return fval, out



# rand('twister',1111)
# A = rand(100,100);
# rand('twister',1112)
# b = rand(200);
# rand('twister',1112)
# b = rand(200,1);
# rand('twister',1113)
# l3 = rand(100,1)*10;
# q,r = qr(A);
#
# [q,r] = qr(A);
# l1 = linspace(1,10,100);
# l2 = linspace(1,1000,100);
# C11 = q.* diag(l1) * q';
# C12 = q.* diag(l3) * q';
# C22 = q.* diag(l2) * q';
# zz = zeros(100,100);
# MM = [C11,C12; zz,C22];
# spy(MM)
# lss = lasso(MM,b,'Lambda',1.e-8);
# plot(lss)
# solvePO
# cvx_begin
# cvx_precision best
# variables x_cvx(200,1)
#
# minimize .5*norm(MM*x_cvx - b) + 1.e-8*sum(abs(x_cvx));
# subject to:
#     x_cvx >= 0;
# cvx_end

