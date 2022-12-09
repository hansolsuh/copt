import warnings
import numpy as np
from numpy.linalg import norm
from numpy import dot
from scipy import optimize, linalg, sparse 
import scipy as sp
import matplotlib.pyplot as plt
from collections import deque

from . import utils

def Hcalc(a_bb1,a_bb2,sk,yk,mu,Hinv,rho,it):

    if rho is not None:
        C_rt = 1+ rho/((it+1)*(1+np.log(it+1)))

    n = Hinv.size
    param = (sk*yk+mu*Hinv)/(sk*sk+mu)
    for i in range(0,n):
        if rho is not None:
            temp1 = max((1/a_bb1),(1/C_rt)*Hinv[i])
            temp2 = min((1/a_bb2),C_rt*Hinv[i])
        else:
            temp1 = 1/a_bb1
            temp2 = 1/a_bb2

        if param[i] < temp1 and a_bb1 >= 0:
            Hinv[i] = temp1
        elif param[i] > temp2 and a_bb2 >=0:
            Hinv[i] = temp2
        else:
            Hinv[i] = param[i]
    return;



def minimize_three_split(
    f_grad,
    x0,
    prox_1=None,
    prox_2=None,
    tol=1e-6,
    max_iter=1000,
    verbose=0,
    callback=None,
    line_search=True,
    step_size=None,
    max_iter_backtracking=100,
    backtracking_factor=0.7,
    h_Lipschitz=None,
    barrier=None,
    hcopt=None,
    Hinv=None,
    total_func=None,
    vm_type=None,
    sbb_n=None,
    sigma=1,
    rho=None,
    args_prox=(),
):

    """Davis-Yin three operator splitting method.

    This algorithm can solve problems of the form

                minimize_x f(x) + g(x) + h(x)

    where f is a smooth function and g and h are (possibly non-smooth)
    functions for which the proximal operator is known.

    Args:
      f_grad: callable
        Returns the function value and gradient of the objective function.
        With return_gradient=False, returns only the function value.

      x0 : array-like
        Initial guess

      prox_1 : callable or None, optional
        prox_1(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha.

      prox_2 : callable or None, optional
        prox_2(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha.

      tol: float, optional
        Tolerance of the stopping criterion.

      max_iter : int, optional
        Maximum number of iterations.

      verbose : int, optional
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

      callback : callable, optional
        Callback function. Takes a single argument (x) with the
        current coefficients in the algorithm. The algorithm will exit if
        callback returns False.

      line_search : boolean, optional
        Whether to perform line-search to estimate the step size.

      step_size : float, optional
        Starting value for the line-search procedure.

      max_iter_backtracking : int, optional
        Maximun number of backtracking iterations.  Used in line search.

      backtracking_factor : float, optional
        The amount to backtrack by during line search.

      args_prox : tuple, optional
        Optional Extra arguments passed to the prox functions.
      
      h_Lipschitz : float, optional
        If given, h is assumed to be Lipschitz continuous with constant h_Lipschitz.

      barrier : float, optional
        Barrier parameter for Interior Point Method. 
        If given, it will decrease when tolerance is met for the certificate

      total_func : float, optional
        Evaluate function value of overall problem.

      vm_type : int, optional
        Sets type of variable metric. 1 is for Barzili-Borwein type, and 2 is for Malitsky-Mishenko type.

      sbb_n : int, optional
        Chooses option for stabilized Barzilai-Borwein Delta. None, or 0, which is default, is min(\|s_1\|,\|s_2\|,\|s_3\|), and
        any other number would be length of queue for min(\|s_i\|,...,\|s_{i-n}\|)

      sigma : float, optional
        Curvature parameter for VM sufficient decrease check.

    Returns:
      res : OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.


    References:
      [1] Davis, Damek, and Wotao Yin. `"A three-operator splitting scheme and
      its optimization applications."
      <https://doi.org/10.1007/s11228-017-0421-z>`_ Set-Valued and Variational
      Analysis, 2017.

      [2] Pedregosa, Fabian, and Gauthier Gidel. `"Adaptive Three Operator
      Splitting." <https://arxiv.org/abs/1804.02339>`_ Proceedings of the 35th
      International Conference on Machine Learning, 2018.
    """
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError("Line search iterations need to be greater than 0")

    if prox_1 is None:

        def prox_1(x, s, *args):
            return x

    if prox_2 is None:

        def prox_2(x, s, *args):
            return x

    x0_temp = np.copy(x0)

    n = x0.size
    VM_trigger = 1

    if Hinv is None:
        Hinv = np.ones(n)
        VM_trigger = 0

    if step_size is None:
        line_search = True
        step_size = 1.0 / utils.init_lipschitz(f_grad, x0)

    if vm_type is None:
        vm_type = 1

    if sbb_n is None:
        sbb_n = 0

    z_old = x0
    z = prox_2(x0, step_size, *args_prox)
    LS_EPS = np.finfo(np.float).eps

    fk, grad_fk = f_grad(z)
    x_old = x0

    x = prox_1(z - step_size * grad_fk, step_size, *args_prox)
    u = np.zeros_like(x)


    delta = 2
    mu    = 10000
    C     = 10
    r     = 1

    aa_list = []
    a1_list = []
    a2_list = []
    Hinv_avglist = []
    Fval_list = []
    M_ls = 5
    b_ls = 1/0.7
    nm_bt =5 #Size of deque for non-monotonic linesearch
    nm_bt_dq = deque(nm_bt*[0],nm_bt)

    bb_stab_delta = np.infty
    norm_sk = np.infty

    if VM_trigger and not line_search:        
        bb_stab_delta_ls_trigger = True
        sx_norm_list = []
    else:
        bb_stab_delta_ls_trigger = False

    theta = np.infty
    Theta = np.infty
    a_bb1 = step_size
    a_bb2 = step_size
    
    for it in range(max_iter):

        grad_fk_old = grad_fk
        fk, grad_fk = f_grad(z)
        nm_bt_dq.append(fk)

        if VM_trigger and it > 1:
#            sk = x - x_old
            sk = z - z_old
            yk = grad_fk - grad_fk_old
            sy = dot(sk,yk)
            ss = dot(sk,sk)
            yy = dot(yk,yk)
            sk_norm = np.linalg.norm(sk)
            yk_norm = np.linalg.norm(yk)

            if it > 2:
                a_bb1_old = a_bb1
                a_bb2_old = a_bb2

            if vm_type == 1:
                a_bb1 = ss/sy
                a_bb2 = sy/yy
            else:
                #MM version
                #a_bb1 = \Lambda
                #a_bb2 = \lambda
                a_bb2 = min(np.sqrt(1+theta/2)*a_bb2, sk_norm/(2*yk_norm))
                a_bb1 = max((1./np.sqrt(1+Theta/2))*a_bb1, (2*sk_norm)/yk_norm)

            if sbb_n == 0:
                if it < 4 and bb_stab_delta_ls_trigger and vm_type == 1:
                    norm_sk = min(norm_sk,np.linalg.norm(sk))
                if bb_stab_delta_ls_trigger and vm_type==1:
                    if it >= 4:
                        bb_stab_delta = 2*norm_sk
                    else: 
                        bb_stab_delta = np.infty
                    a_bb2 = min(a_bb2, bb_stab_delta/np.linalg.norm(grad_fk))
            else:
                if bb_stab_delta_ls_trigger and vm_type == 1:
                    if len(sx_norm_list) >= sbb_n and len(sx_norm_list) != 0:
                        sx_norm_list.pop(0)
                    sx_norm_list.append(np.linalg.norm(sk))
                    if it <= 5:
                        bb_stab_delta = np.infty
                    else:
                        bb_stab_delta = 2*min(sx_norm_list)
                    a_bb2 = min(a_bb2, bb_stab_delta/np.linalg.norm(grad_fk))



            if a_bb1 < 0:
                a_bb1 = a_bb1_old
            if a_bb2 < 0:
                a_bb2 = a_bb2_old

            if vm_type == 2 and it > 2:
                theta = a_bb2/a_bb2_old
                Theta = a_bb1/a_bb1_old

            a1_list.append(a_bb1)
            a2_list.append(a_bb2)
            Hcalc(a_bb1,a_bb2,sk,yk,mu,Hinv,rho,it)
            Hinv_avglist.append(np.average(Hinv))

        x_old = x

        if VM_trigger and it > 1:
            x = prox_1(z - (1/Hinv) *  (u + grad_fk), 1/Hinv, *args_prox)
        else:
            x = prox_1(z - step_size *  (u + grad_fk), step_size, *args_prox)
        
        incr = x - z
        norm_incr = np.linalg.norm(incr)
    
        fx = f_grad(x, return_gradient=False) 
        ls = norm_incr > 1e-7 and line_search
        if ls:
            for it_ls in range(max_iter_backtracking):
                if VM_trigger and it > 1:
                    tmp = sigma*incr.dot(incr*Hinv)
                    rhs = np.max(nm_bt_dq) + grad_fk.dot(incr) + 0.5*tmp
                else:
                    rhs = fk + grad_fk.dot(incr) + (norm_incr ** 2) / (2 * step_size)
                ls_tol = fx - rhs
                if ls_tol <= 1.e-12:
                    # step size found
                    # if ls_tol > 0:
                    #     ls_tol = 0.
                    break
                else:
                    if VM_trigger and it>1:
                        Hinv /= backtracking_factor

                    else:
                        step_size *= backtracking_factor

        z_old = z
        if VM_trigger and it > 1:
            z = prox_2(x + (1/Hinv)* u, 1/Hinv, *args_prox)
            u += (x - z) / (1/Hinv)
        else:
            z = prox_2(x + step_size * u, step_size, *args_prox)
            u += (x - z) / step_size


        if VM_trigger:
            certificate = norm_incr * Hinv
            certificate = np.max(certificate)
        else:
            certificate = norm_incr / step_size

        if ls and h_Lipschitz is not None:
            if h_Lipschitz == 0:
                step_size = step_size * 1.02
            else:
                quot = h_Lipschitz ** 2
                tmp = np.sqrt(step_size ** 2 + (2 * step_size / quot) * (-ls_tol))
                step_size = min(tmp, step_size * 1.02)

        if callback is not None:
            if callback(locals()) is False:
                break
   
        if it > 0 and certificate < tol:
            #TODO barrier with callback func while using the logger??
            if barrier != None:
                if barrier > 1.e-12:
                    barrier /= 1.1
                else:
                    success = True
                    break
            else:                
                success = True
                break
    
    return optimize.OptimizeResult(
        x=x, success=success, nit=it, certificate=certificate, step_size=step_size
    )


def minimize_primal_dual(
    f_grad,
    x0,
    prox_1=None,
    prox_2=None,
    L=None,
    tol=1e-12,
    max_iter=1000,
    callback=None,
    step_size=1.0,
    step_size2=None,
    line_search=True,
    max_iter_ls=20,
    barrier=None,
    verbose=0,
):
    """Primal-dual hybrid gradient splitting method.

    This method for optimization problems of the form

            minimize_x f(x) + g(x) + h(L x)

    where f is a smooth function and g is a (possibly non-smooth)
    function for which the proximal operator is known.

    Args:
      f_grad: callable
          Returns the function value and gradient of the objective function.
          It should accept the optional argument return_gradient, and when False
          it should return only the function value.

      prox_1 : callable of the form prox_1(x, alpha)
          prox_1(x, alpha, *args) returns the proximal operator of g at x
          with parameter alpha.

      prox_2 : callable or None
          prox_2(y, alpha, *args) returns the proximal operator of h at y
          with parameter alpha.

      x0 : array-like
          Initial guess of solution.

      L : array-like or linear operator
          Linear operator inside the h term. It may be any of the following types:
             - ndarray
             - matrix
             - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
             - LinearOperator
             - An object with .shape and .matvec attributes

      max_iter : int
          Maximum number of iterations.

      verbose : int
          Verbosity level, from 0 (no output) to 2 (output on each iteration)

      callback : callable.
          callback function (optional). Takes a single argument (x) with the
          current coefficients in the algorithm. The algorithm will exit if
          callback returns False.

    Returns:
      res : OptimizeResult
          The optimization result represented as a
          ``scipy.optimize.OptimizeResult`` object. Important attributes are:
          ``x`` the solution array, ``success`` a Boolean flag indicating if
          the optimizer exited successfully and ``message`` which describes
          the cause of the termination. See `scipy.optimize.OptimizeResult`
          for a description of other attributes.

    References:

        * Malitsky, Yura, and Thomas Pock. `A first-order primal-dual algorithm with linesearch <https://arxiv.org/pdf/1608.08883.pdf>`_,
        SIAM Journal on Optimization (2018) (Algorithm 4 for the line-search variant)

        * Condat, Laurent. "A primal-dual splitting method for convex optimization
        involving Lipschitzian, proximable and linear composite terms." Journal of
        Optimization Theory and Applications (2013).
    """
    x = np.array(x0, copy=True)
    n_features = x.size

    if L is None:
        L = sparse.eye(n_features, n_features, format="csr")
    L = sparse.linalg.aslinearoperator(L)

    y = L.matvec(x)

    success = False
    if not max_iter_ls > 0:
        raise ValueError("Line search iterations need to be greater than 0")

    if prox_1 is None:

        def prox_1(x, step_size):
            return x

    if prox_2 is None:

        def prox_2(x, step_size):
            return x

    # conjugate of prox_2
    def prox_2_conj(x, ss):
        return x - ss * prox_2(x / ss, 1.0 / ss)

    # .. main iteration ..
    theta = 1.0
    delta = 0.5
    sigma = step_size
    if step_size2 is None:
        ss_ratio = 0.5
        tau = ss_ratio * sigma
    else:
        tau = step_size2
        ss_ratio = tau / sigma

    fk, grad_fk = f_grad(x)
    norm_incr = np.infty
    x_next = x.copy()

    for it in range(max_iter):
        y_next = prox_2_conj(y + tau * L.matvec(x), tau)
        if line_search:
            tau_next = tau * (1 + np.sqrt(1 + theta)) / 2
            while True:
                theta = tau_next / tau
                sigma = ss_ratio * tau_next
                y_bar = y_next + theta * (y_next - y)
                x_next = prox_1(x - sigma * (L.rmatvec(y_bar) + grad_fk), sigma)
                incr_x = np.linalg.norm(L.matvec(x_next) - L.matvec(x))
                f_next, f_grad_next = f_grad(x_next)
                if incr_x <= 1e-10:
                    break

                tmp = (sigma * tau_next) * (incr_x ** 2)
                tmp += 2 * sigma * (f_next - fk - grad_fk.dot(x_next - x))
                if tmp / delta <= (incr_x ** 2):
                    tau = tau_next
                    break
                else:
                    tau_next *= 0.9
        else:
            y_bar = 2 * y_next - y
            x_next = prox_1(x - sigma * (L.rmatvec(y_bar) + grad_fk), sigma)
            f_next, f_grad_next = f_grad(x_next)

        if it % 100 == 0:
            norm_incr = linalg.norm(x_next - x) + linalg.norm(y_next - y)

        x[:] = x_next[:]
        y[:] = y_next[:]
        fk, grad_fk = f_next, f_grad_next

        if norm_incr < tol:
            if barrier != None:
                if barrier > 1.e-12:
                    barrier /= 1.1
            else:
                success = True
                break

        if callback is not None:
            if callback(locals()) is False:
                break
    if it >= max_iter:
        warnings.warn(
            "proximal_gradient did not reach the desired tolerance level",
            RuntimeWarning,
        )

    return optimize.OptimizeResult(
        x=x, success=success, nit=it, certificate=norm_incr, step_size=sigma
    )
