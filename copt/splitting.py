import warnings
import numpy as np
from numpy.linalg import norm
from numpy import dot
from scipy import optimize, linalg, sparse
import matplotlib.pyplot as plt

from . import utils

def Hcalc(a_bb1,a_bb2,sk,yk,mu,Hinv):
#    import pdb
#    pdb.set_trace()
    n = Hinv.size
    param = (sk*yk+mu*Hinv)/(yk*yk+mu)
    for i in range(0,n):
        if param[i] < a_bb1:
            Hinv[i] = a_bb1
        elif param[i] > a_bb2:
            Hinv[i] = a_bb2
        else:
            Hinv[i] = param[i]
#    import pdb
#    pdb.set_trace()
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
    L_Lip=None,
    max_iter_backtracking=100,
    backtracking_factor=0.7,
    h_Lipschitz=None,
    barrier=None,
    hcopt=None,
    Hinv=None,
    total_func=None,
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

    n = x0.size
    VM_trigger = 1

    if Hinv is None:
        Hinv = np.ones(n)
        VM_trigger = 0

    if step_size is None:
        line_search = True
        step_size = 1.0 / utils.init_lipschitz(f_grad, x0)

    import pdb
    pdb.set_trace()
    z_old = x0
    z = prox_2(x0, step_size, *args_prox)
    LS_EPS = np.finfo(np.float).eps

    fk, grad_fk = f_grad(z)
    x_old = x0
    x = prox_1(z - step_size * grad_fk, step_size, *args_prox)
    u = np.zeros_like(x)

    delta = 2
    mu    = 1000000 #TODO large mu for static hessian. how large?
    C     = 10
    r     = 1

    aa_list = []
    a1_list = []
    a2_list = []
    Hinv_avglist = []
    Fval_list = []
    M_ls = 5
    b_ls = 1.1

    for it in range(max_iter):
#            import pdb
#            pdb.set_trace()

        grad_fk_old = grad_fk
        fk, grad_fk = f_grad(z)


        if VM_trigger and it > 1: 
            sk = x - x_old
            yk = grad_fk - grad_fk_old #TODO grad_f(x) not z?
            hk = C + np.max(-dot(sk,yk)/(norm(sk)**2),0)*(norm(grad_fk_old)**(-r))

            if it > 0:
                a_bb1 = dot(yk,yk)/(2*dot(sk,sk))
                a_bb2 = dot(yk,yk)/dot(sk,yk)
                a1_list.append(a_bb1)
                a2_list.append(a_bb2)
                Hcalc(a_bb1,a_bb2,sk,yk,mu,Hinv)
##                Hinv_avglist.append(np.average(Hinv))
        
#        if L_Lip is not None:
#            Hinv[Hinv<L_Lip] = L_Lip
#
#        x_old = x
#        x = prox_1(z - step_size *  (u + Hinv*grad_fk), step_size, *args_prox)
#        total_eval = total_func(x)
#
#        ls_it = 0
#
#        if len(Fval_list) < M_ls:
#            Fval_list.append(total_eval)
#        else:
#            del Fval_list[0]
#            Fval_list.append(total_eval)
#
#        while True and it>1 and VM_trigger: #TODO make breakable? trigger?
#            Hinv *= b_ls
#            x = prox_1(z - step_size *  (u + Hinv*grad_fk), step_size, *args_prox)
#            total_eval = total_func(x)
#
#            
#            xdiff = x-x_old
##            if total_eval <= max(Fval_list) - dot(Hinv*xdiff, xdiff):
#            if total_eval <= max(Fval_list) - dot(Hinv*xdiff, xdiff):
#                print(ls_it)
#                break
#            ls_it += 1
#
#            if ls_it > 1000:
##                Hinv.fill(1)
##                VM_trigger = None
##                print(it)
#                break


#
##        if it == 100 and VM_trigger:
##            Hinv.fill(1)
##            VM_trigger = None

        incr = x - z
        norm_incr = np.linalg.norm(incr)
        ls = norm_incr > 1e-7 and line_search
        if ls:
            for it_ls in range(max_iter_backtracking):
                rhs = fk + grad_fk.dot(incr) + (norm_incr ** 2) / (2 * step_size)
                ls_tol = f_grad(x, return_gradient=False) - rhs
                if ls_tol <= LS_EPS:
                    # step size found
                    # if ls_tol > 0:
                    #     ls_tol = 0.
                    break
                else:
                    step_size *= backtracking_factor

        z_old = z
        z = prox_2(x + step_size * u, step_size, *args_prox)
        u += (x - z) / step_size
        certificate = norm_incr / step_size


#        if total_func is not None:
#            fvvv = total_func(x)
#            Fval_list.append(fvvv)


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
            if barrier != None:
                if barrier > 1.e-12:
                    barrier /= 1.1
            else:                
                success = True
                break
            
#    plt.plot(temp_list)
#    plt.show()
#    import pdb
#    pdb.set_trace()
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
