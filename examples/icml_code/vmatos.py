import warnings
import portfoliodata
import sparse_nuclear_norm_data
import overlapping_group_lasso
import tomodata
import sampledata
import csv
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
import scipy as sp
from scipy import sparse,optimize
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import dsolve
from scipy.sparse import hstack, vstack
from scipy.optimize import line_search
from scipy.optimize import minimize
import copt as cp
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import gc
import math
import sys
import time


if __name__ == "__main__":
    #Flag1 : run Tomo toggle
    #Flag2 : run TV toggle
    #Flag3 : run sparse norm toggle
    #Flag4 : run Lasso toggle

    flag1 = 1
    flag2 = 1
    flag3 = 1
    flag4 = 1

    flag1 = sys.argv[1]
    flag2 = sys.argv[2]
    flag3 = sys.argv[3]
    flag4 = sys.argv[4]
    
    max_iter = 2000
    if flag1 == '1':
        #Tomography Data
        F,b,GT        = tomodata.generate_data()
        Fb            = F.T @ b
        [m,n]         = F.shape
        Hinv1         = np.ones(n)
        e             = np.ones(n)
        f_copt        = tomodata.f3_copt(F,b,Fb)
        f_grad        = f_copt.f_grad
        tomo_stepsize = 2/0.00015339534146545738
        lbd           = [1.e-8]
    
        tomo_VM_LS_obj,  tomo_AVM_obj,  tomo_AVMN_obj,  tomo_A_obj,  tomo_TOS_obj  = [],[],[],[],[]
        tomo_VM_LS_time, tomo_AVM_time, tomo_AVMN_time, tomo_A_time, tomo_TOS_time = [],[],[],[],[]
    
        out_tomo = []

        f,ax = plt.subplots(1,len(lbd), sharey=False)
        sigma = 0.99
        rho=1
        for i, lbd in enumerate(lbd):

            #1. VM : BB LS version
            mu     = np.array(1.e-6)
            Hinv1  = np.ones(n)
            h_copt = tomodata.f2_copt(mu,Hinv1)
            g_copt = tomodata.f1_copt(e,lbd,Hinv1)
    
            cb_adatos_vm_ls = cp.utils.Trace()
            adatos_avm_ls   = cp.minimize_three_split(f_grad,np.zeros(n), h_copt.prox,g_copt.prox,step_size=tomo_stepsize, max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,barrier=mu,Hinv=Hinv1,line_search=True, #stablized BB
                           vm_type=1,callback=cb_adatos_vm_ls, sigma=sigma,rho=rho)
    
            trace_obj = None
            trace_obj = [norm(x-GT)/norm(GT) for x in cb_adatos_vm_ls.trace_x]
            tomo_VM_LS_obj.append(trace_obj)
            tomo_VM_LS_time.append(cb_adatos_vm_ls.trace_time)
            out_tomo.append(adatos_avm_ls.x.reshape(int(math.sqrt(n)),int(math.sqrt(n))))
   
            #2. VM : BB3 version
            mu     = np.array(1.e-6)
            Hinv1  = np.ones(n)
            h_copt = tomodata.f2_copt(mu,Hinv1)
            g_copt = tomodata.f1_copt(e,lbd,Hinv1)
    
            cb_adatos_vm = cp.utils.Trace()
            adatos_avm   = cp.minimize_three_split(f_grad,np.zeros(n), h_copt.prox,g_copt.prox,step_size=tomo_stepsize, max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,barrier=mu,Hinv=Hinv1,line_search=False, #stablized BB
                  vm_type=1,callback=cb_adatos_vm, sigma=sigma,rho=rho)
    
            trace_obj = None
            trace_obj = [norm(x-GT)/norm(GT) for x in cb_adatos_vm.trace_x]
            tomo_AVM_obj.append(trace_obj)
            tomo_AVM_time.append(cb_adatos_vm.trace_time)
            out_tomo.append(adatos_avm.x.reshape(int(math.sqrt(n)),int(math.sqrt(n))))
   
            #3. VM : BB-N version
            mu     = np.array(1.e-6)
            Hinv1  = np.ones(n)
            h_copt = tomodata.f2_copt(mu,Hinv1)
            g_copt = tomodata.f1_copt(e,lbd,Hinv1)
    
            cb_adatos_vmn = cp.utils.Trace()
            adatos_avmn   = cp.minimize_three_split(f_grad,np.zeros(n), h_copt.prox,g_copt.prox,step_size=tomo_stepsize, max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,barrier=mu,Hinv=Hinv1,line_search=False, #stablized BB
                          vm_type=1,sbb_n=10,callback=cb_adatos_vmn, sigma=sigma,rho=rho)
    
            trace_obj = None
            trace_obj = [norm(x-GT)/norm(GT) for x in cb_adatos_vmn.trace_x]
            tomo_AVMN_obj.append(trace_obj)
            tomo_AVMN_time.append(cb_adatos_vmn.trace_time)
            out_tomo.append(adatos_avmn.x.reshape(int(math.sqrt(n)),int(math.sqrt(n))))

            #4. A : ATOS without variable metric (ie, Identity for Hinv)
            mu = np.array(1.e-6)
            h_copt = tomodata.f2_copt(mu,1)
            g_copt = tomodata.f1_copt(e,lbd,1)
    
            cb_adatos = cp.utils.Trace()
            adatos_a  = cp.minimize_three_split(f_grad,np.zeros(n), h_copt.prox,g_copt.prox,step_size=tomo_stepsize, max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,barrier=mu,callback=cb_adatos)
    
            trace_obj = None
            trace_obj = [norm(x-GT)/norm(GT) for x in cb_adatos.trace_x]
            tomo_A_obj.append(trace_obj)
            tomo_A_time.append(cb_adatos.trace_time)
            out_tomo.append(adatos_a.x.reshape(int(math.sqrt(n)),int(math.sqrt(n))))

            #5. TOS : TOS without variable metric (ie, Identity for Hinv)
            mu     = np.array(1.e-6)
            h_copt = tomodata.f2_copt(mu,1)
            g_copt = tomodata.f1_copt(e,lbd,1)
    
            cb_tos = cp.utils.Trace()
            adatos_tos = cp.minimize_three_split(f_grad,np.zeros(n), h_copt.prox,g_copt.prox,step_size=tomo_stepsize, max_iter=max_iter, tol=1.e-6,barrier=mu,callback=cb_tos,line_search=False)
    
            trace_obj = None
            trace_obj = [norm(x-GT)/norm(GT) for x in cb_tos.trace_x]
            tomo_TOS_obj.append(trace_obj)
            tomo_TOS_time.append(cb_tos.trace_time)
            out_tomo.append(adatos_tos.x.reshape(int(math.sqrt(n)),int(math.sqrt(n))))

            ax.set_title("Tomography Example $\lambda=%s$" % lbd)
            ax.plot(tomo_VM_LS_obj[i], label='VM-BB-LS')
            ax.plot(tomo_AVM_obj[i], label='VM-SBB3')
            ax.plot(tomo_AVMN_obj[i], label='VM-SBBN')
            ax.plot(tomo_A_obj[i], label='A')
            ax.plot(tomo_TOS_obj[i], label='TOS')
            ax.set_xlim(left=0)
            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Relative Error')
            ax.legend()

        plt.show()

    #Overlapping Lasso
    if flag2 == '1':
        A_ls,b_ls,f_ls,groups_ls,GT_ls,n_features_ls,ls_stepsize = overlapping_group_lasso.generate_data()

        all_betas = np.logspace(-4, -3, 2)
        ls_AVM_obj,  ls_A_obj,  ls_VM_obj,  ls_TOS_obj,  = [],[],[],[]
        ls_AVM_time, ls_A_time, ls_VM_time, ls_TOS_time, = [],[],[],[]
    
        out_ls = []

        Hinv = np.ones(n_features_ls)
        e = np.ones(n_features_ls)
        Lip = ls_stepsize*0.99

        f,ax = plt.subplots(1, len(all_betas), sharey=False)

        sigma = 0.99
        rho = 10000
        for i, beta in enumerate(all_betas):
            print('beta = %s' % beta)

            #1. VM - BB method
            Hinv = np.ones(n_features_ls)
            G1 = cp.penalty.GroupL1(beta, groups_ls[::2] )
            G2 = cp.penalty.GroupL1(beta, groups_ls[1::2] )
            cb_adatos_vm = cp.utils.Trace()
            adatos_avm   = cp.minimize_three_split(f_ls.f_grad,np.zeros(n_features_ls), G1.prox,G2.prox,step_size=10*ls_stepsize,max_iter=max_iter, tol=1.e-6,h_Lipschitz=beta,line_search=False,
                                                   vm_type=1,Hinv=Hinv,callback=cb_adatos_vm, sigma=sigma, rho=rho)

            trace_obj = None
            trace_obj = [overlapping_group_lasso.loss(x, f_ls, G1, G2) for x in cb_adatos_vm.trace_x]
            ls_AVM_obj.append(trace_obj)
            ls_AVM_time.append(cb_adatos_vm.trace_time)
            out_ls.append(adatos_avm.x)

            #2. A : ATOS without variable metric (ie, Identity for Hinv)
            G1 = cp.penalty.GroupL1(beta, groups_ls[::2])
            G2 = cp.penalty.GroupL1(beta, groups_ls[1::2])
            cb_adatos = cp.utils.Trace()
            adatos_a  = cp.minimize_three_split(f_ls.f_grad,np.zeros(n_features_ls), G1.prox,G2.prox,step_size=10*ls_stepsize, max_iter=max_iter,tol=1.e-6,h_Lipschitz=beta,callback=cb_adatos)
    
            trace_obj = None
            trace_obj = [overlapping_group_lasso.loss(x, f_ls, G1, G2) for x in cb_adatos.trace_x]
            ls_A_obj.append(trace_obj)
            ls_A_time.append(cb_adatos.trace_time)
            out_ls.append(adatos_a.x)
    
            #3. TOS : TOS without variable metric (ie, Identity for Hinv)
            G1 = cp.penalty.GroupL1(beta, groups_ls[::2])
            G2 = cp.penalty.GroupL1(beta, groups_ls[1::2])
            cb_tos = cp.utils.Trace()
            adatos_tos = cp.minimize_three_split(f_ls.f_grad,np.zeros(n_features_ls), G1.prox,G2.prox,step_size=ls_stepsize, max_iter=max_iter, tol=1.e-6,callback=cb_tos,line_search=False)
    
            trace_obj = None
            trace_obj = [overlapping_group_lasso.loss(x, f_ls, G1, G2) for x in cb_tos.trace_x]
            ls_TOS_obj.append(trace_obj)
            ls_TOS_time.append(cb_tos.trace_time)
            out_ls.append(adatos_tos.x)
    
            ax[i].set_title("Group Lasso $beta=%s$" % beta)
            ax[i].plot( ls_AVM_obj[i], label='VM-SBB')
            ax[i].plot(ls_A_obj[i], label='A')
            ax[i].plot(ls_TOS_obj[i], label='TOS')
            ax[i].set_yscale('log')
            ax[i].set_xlim(left=0)
            ax[i].set_xlabel('Iteration')
            ax[i].set_ylabel('Objective')
            ax[i].legend()

        plt.show()

    #Anisotropic Hessian. Data randomly generated
    #Trying to solve 0.5\|Fx-b\|_2^2 + lambda*|x|_1 + logbarrier
    #F = [C11,C12; 0, C22]
    # |C11| = k1
    # |C22| = k2
    if flag3 == '1':
        sample_VM_LS_obj,  sample_AVM_obj,  sample_AVMN_obj,  sample_A_obj,  sample_VM_obj,  sample_TOS_obj  = [],[],[],[],[],[]
        sample_VM_LS_time, sample_AVM_time, sample_AVMN_time, sample_A_time, sample_VM_time, sample_TOS_time = [],[],[],[],[],[]

        out_sample = []

        n  = 100
        k1 = 10
        k2 = 10000
        Hinv = np.ones(2*n)
        mu = np.array(1.e-6)
        e = np.ones(2*n)
        lbd = 1.e-8
        max_iter = 500

        f,ax = plt.subplots(1, 1, sharey=False)

        sigma = 0.99
        rho=1
        #1. VM - BB-LS
        mu = np.array(1.e-6)
        f_opt = sampledata.f_copt(n,k1,k2)
        g_opt = tomodata.f1_copt(e,lbd,Hinv)
        h_opt = tomodata.f2_copt(mu,Hinv)
        step_size = 2/f_opt.Lip()
        cb_adatos_vm_ls = cp.utils.Trace()
        adatos_avm_ls   = cp.minimize_three_split(f_opt.f_grad,np.zeros(2*n), h_opt.prox,g_opt.prox,step_size=step_size,max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,line_search=True,
                                           vm_type=1,barrier=mu,Hinv= Hinv,callback=cb_adatos_vm_ls, sigma=sigma,rho=rho)

        trace_obj = None
        trace_obj = [f_opt(x) + g_opt(x) for x in cb_adatos_vm_ls.trace_x]
        sample_VM_LS_obj.append(trace_obj)
        sample_VM_LS_time.append(cb_adatos_vm_ls.trace_time)
        out_sample.append(adatos_avm_ls.x)

        #2. VM - BB-3
        mu = np.array(1.e-6)
        Hinv = np.ones(2*n)
        f_opt = sampledata.f_copt(n,k1,k2)
        g_opt = tomodata.f1_copt(e,lbd,Hinv)
        h_opt = tomodata.f2_copt(mu,Hinv)
        step_size = 2/f_opt.Lip()
        cb_adatos_vm = cp.utils.Trace()
        adatos_avm   = cp.minimize_three_split(f_opt.f_grad,np.zeros(2*n), h_opt.prox,g_opt.prox,step_size=step_size,max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,line_search=False, #Stab BB
                                               vm_type=1,barrier=mu,Hinv= Hinv,callback=cb_adatos_vm, sigma=sigma,rho=rho)

        trace_obj = None
        trace_obj = [f_opt(x) + g_opt(x) for x in cb_adatos_vm.trace_x]
        sample_AVM_obj.append(trace_obj)
        sample_AVM_time.append(cb_adatos_vm.trace_time)
        out_sample.append(adatos_avm.x)

        #3. VM - BB-N
        mu = np.array(1.e-6)
        Hinv = np.ones(2*n)
        f_opt = sampledata.f_copt(n,k1,k2)
        g_opt = tomodata.f1_copt(e,lbd,Hinv)
        h_opt = tomodata.f2_copt(mu,Hinv)
        cb_adatos_vmn = cp.utils.Trace()
        adatos_avmn   = cp.minimize_three_split(f_opt.f_grad,np.zeros(2*n), h_opt.prox,g_opt.prox,step_size=step_size,max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,line_search=False, #Stab BB
                                              vm_type=1,barrier=mu,Hinv= Hinv,sbb_n=10,callback=cb_adatos_vmn, sigma=sigma,rho=rho)

        trace_obj = None
        trace_obj = [f_opt(x) + g_opt(x) for x in cb_adatos_vmn.trace_x]
        sample_AVMN_obj.append(trace_obj)
        sample_AVMN_time.append(cb_adatos_vmn.trace_time)
        out_sample.append(adatos_avmn.x)
    
    
        #4. A : ATOS
        mu    = np.array(1.e-6)
        g_opt = tomodata.f1_copt(e,lbd,1)
        h_opt = tomodata.f2_copt(mu,1)
        step_size = 2/f_opt.Lip()
        cb_adatos = cp.utils.Trace()
        adatos_a  = cp.minimize_three_split(f_opt.f_grad,np.zeros(2*n), h_opt.prox,g_opt.prox,step_size=step_size,max_iter=max_iter, tol=1.e-6,h_Lipschitz=lbd,barrier=mu,callback=cb_adatos)
    
        trace_obj = None
        trace_obj = [f_opt(x) + g_opt(x) for x in cb_adatos.trace_x]
        sample_A_obj.append(trace_obj)
        sample_A_time.append(cb_adatos.trace_time)
        out_sample.append(adatos_a.x)
    
        #5. TOS : TOS
        mu    = np.array(1.e-6)
        g_opt = tomodata.f1_copt(e,lbd,1)
        h_opt = sampledata.h_copt(mu,1)
        step_size = 2/f_opt.Lip()

        cb_tos = cp.utils.Trace()
        adatos_tos = cp.minimize_three_split(f_opt.f_grad,np.zeros(2*n), h_opt.prox,g_opt.prox,step_size=step_size,max_iter=max_iter, tol=1.e-6,barrier=mu,callback=cb_tos,line_search=False)
    
        trace_obj = None
        trace_obj = [f_opt(x) + g_opt(x) for x in cb_tos.trace_x]
        sample_TOS_obj.append(trace_obj)
        sample_TOS_time.append(cb_tos.trace_time)
        out_sample.append(adatos_tos.x)

        ax.set_title('Anisotropic Example')
        ax.plot(sample_VM_LS_obj[0], label='VM-BB-LS')
        ax.plot(sample_AVM_obj[0], label='VM-SBB3')
        ax.plot(sample_AVMN_obj[0], label='VM-SBBN')
        ax.plot(sample_A_obj[0], label='A')
        ax.plot(sample_TOS_obj[0], label='TOS')
        ax.set_yscale('log')
        ax.set_xlim(left=0,right=200)
        ax.set_ylim(top=22,bottom=16.8)
        ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        plt.show()


    #Portfolio Optimization
    if flag4 == '1':
        Sigma, port_r, port_m, port_gt, port_eval  = portfoliodata.generate_data()

        all_betas = np.logspace(-4, -2, 3)
        port_AVM_obj,  port_A_obj,  port_TOS_obj,  port_APDHG_obj,  port_PDHG_obj  = [],[],[],[],[]
        port_AVM_time, port_A_time, port_TOS_time, port_APDHG_time, port_PDHG_time = [],[],[],[],[]
    
        out_port = []
        port_gt_norm = np.linalg.norm(port_gt)

        n_features_port = 1000
        Hinv = np.ones(n_features_port)
        e = np.ones(n_features_port)
        f_copt        = portfoliodata.f1_copt(Hinv,Sigma)
        f_grad        = f_copt.f_grad

        f,ax = plt.subplots(1, 1, sharey=False)
        port_stepsize = 4.013 #1.99*(1/norm(sigma))
        sigma=0.99
        rho=1

        Hinv = np.ones(n_features_port)
        G1 = portfoliodata.f2_copt(Hinv)
        G2 = portfoliodata.f3_copt(Hinv,port_r,port_m)
        
        #1. VM-BB
        cb_adatos_vm = cp.utils.Trace()
        adatos_avm   = cp.minimize_three_split(f_copt.f_grad,np.zeros(n_features_port), G1.prox,G2.prox,step_size=port_stepsize,max_iter=max_iter, tol=1.e-6,line_search=False, #Stab BB
                                            vm_type=1,Hinv=Hinv,total_func = f_copt,callback=cb_adatos_vm, sigma=sigma,rho=rho)

        trace_obj = None
        trace_obj = [np.linalg.norm(x-port_gt)/port_gt_norm for x in cb_adatos_vm.trace_x]
        port_AVM_obj.append(trace_obj)
        port_AVM_time.append(cb_adatos_vm.trace_time)
        out_port.append(adatos_avm.x)
        
        #2. A 
        G1 = portfoliodata.f2_copt(1)
        G2 = portfoliodata.f3_copt(1,port_r,port_m)
        cb_atos = cp.utils.Trace()
        adatos_atos = cp.minimize_three_split(f_copt.f_grad,np.zeros(n_features_port), G1.prox,G2.prox,step_size=port_stepsize, max_iter=max_iter,tol=1.e-6,callback=cb_atos)
        trace_obj = None
        trace_obj = [np.linalg.norm(x-port_gt)/port_gt_norm for x in cb_atos.trace_x]
        port_A_obj.append(trace_obj)
        port_A_time.append(cb_atos.trace_time)
        out_port.append(adatos_atos.x)

        #3. TOS 
        G1 = portfoliodata.f2_copt(1)
        G2 = portfoliodata.f3_copt(1,port_r,port_m)
        cb_tos = cp.utils.Trace()
        adatos_tos = cp.minimize_three_split(f_copt.f_grad,np.zeros(n_features_port), G1.prox,G2.prox,step_size=port_stepsize, max_iter=max_iter,tol=1.e-6,callback=cb_tos,line_search=False)
        trace_obj = None
        trace_obj = [np.linalg.norm(x-port_gt)/port_gt_norm for x in cb_tos.trace_x]
        port_TOS_obj.append(trace_obj)
        port_TOS_time.append(cb_tos.trace_time)
        out_port.append(adatos_tos.x)

        ax.set_title('Portfolio Example')
        ax.plot(port_AVM_obj[0], label='VM-SBB')
        ax.plot(port_A_obj[0], label='A')
        ax.plot(port_TOS_obj[0], label='TOS')
        ax.set_yscale('log')
        ax.set_xlim(left=0)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Error')
        ax.legend()

        plt.show()
