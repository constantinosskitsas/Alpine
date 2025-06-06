# -*- coding: utf-8 -*-
"""
Rewrite ot.bregman.sinkhorn in Python Optimal Transport (https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn)
using pytorch operations.
Bregman projections for regularized OT (Sinkhorn distance).
"""
import torch
import time
import warnings
from memory_profiler import profile

#import numpy as np
M_EPS = 1e-16

def greenkhorn(a, b, M, reg, numItermax=10000, stopThr=1e-9, verbose=False,
               log=False, warn=True, warmstart=None):
    """
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The algorithm used is based on the paper :ref:`[22] <references-greenkhorn>`
    which is a stochastic version of the Sinkhorn-Knopp
    algorithm :ref:`[2] <references-greenkhorn>`

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.bregman.greenkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-greenkhorn:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013

    .. [22] J. Altschuler, J.Weed, P. Rigollet : Near-linear time
        approximation algorithms for optimal transport via Sinkhorn
        iteration, Advances in Neural Information Processing
        Systems (NIPS) 31, 2017


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    #a, b, M = list_to_array(a, b, M)

    #nx = get_backend(M, a, b)
    if nx.__name__ in ("jax", "tf"):
        raise TypeError("JAX or TF arrays have been received. Greenkhorn is not "
                        "compatible with  neither JAX nor TF")

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    K = nx.exp(-M / reg)

    if warmstart is None:
        u = nx.full((dim_a,), 1. / dim_a, type_as=K)
        v = nx.full((dim_b,), 1. / dim_b, type_as=K)
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])
    G = u[:, None] * K * v[None, :]

    viol = nx.sum(G, axis=1) - a
    viol_2 = nx.sum(G, axis=0) - b
    stopThr_val = 1
    if log:
        log = dict()
        log['u'] = u
        log['v'] = v

    for ii in range(numItermax):
        i_1 = nx.argmax(nx.abs(viol))
        i_2 = nx.argmax(nx.abs(viol_2))
        m_viol_1 = nx.abs(viol[i_1])
        m_viol_2 = nx.abs(viol_2[i_2])
        stopThr_val = nx.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            new_u = a[i_1] / nx.dot(K[i_1, :], v)
            G[i_1, :] = new_u * K[i_1, :] * v

            viol[i_1] = nx.dot(new_u * K[i_1, :], v) - a[i_1]
            viol_2 += (K[i_1, :].T * (new_u - old_u) * v)
            u[i_1] = new_u
        else:
            old_v = v[i_2]
            new_v = b[i_2] / nx.dot(K[:, i_2].T, u)
            G[:, i_2] = u * K[:, i_2] * new_v
            # aviol = (G@one_m - a)
            # aviol_2 = (G.T@one_n - b)
            viol += (-old_v + new_v) * K[:, i_2] * u
            viol_2[i_2] = new_v * nx.dot(K[:, i_2], u) - b[i_2]
            v[i_2] = new_v

        if stopThr_val <= stopThr:
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    if log:
        log["n_iter"] = ii
        log['u'] = u
        log['v'] = v

    if log:
        return G, log
    else:
        return G

def sinkhorn(a, b, C, reg=1e-1, method='sinkhorn', maxIter=1000, tau=1e3,
             stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    method : str
        method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    if method.lower() == 'sinkhorn':
        #print('ok mphke')
        return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, C, reg, maxIter=maxIter, tau=tau,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, C, reg,
                                        maxIter=maxIter, maxInnerIter=100, tau=tau,
                                        scaling_base=0.75, scaling_coef=None, stopThr=stopThr,
                                        verbose=False, log=log, warm_start=warm_start, eval_freq=eval_freq,
                                        print_freq=print_freq, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   verbose=True, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """
    device = a.device
    na, nb = C.shape
    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    # t1 = time.time()
    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv)

        if torch.any(KTu == 0) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break
        if  it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            #b_hat = torch.matmul(u, K) * v
            #a_hat = torch.matmul(v, K.T) * u
            #err = (b - b_hat).pow(2).sum().item()

            #a_hat = torch.matmul(v, K.T) * u
            #err = (a[:-1] - a_hat[:-1]).pow(2).sum().item()
            #print(len(a),' ',len(b))
            #divA=(torch.sum(a)/(torch.sum(a)+torch.sum(b)))
            #divB=(torch.sum(b)/(torch.sum(a)+torch.sum(b)))

            if len(a)!=len(b):
                #print('ok')
                b_hat = torch.matmul(u, K) * v
                a_hat = torch.matmul(v, K.T) * u
                err = (a - a_hat).pow(2).sum().item()/2 + (b - b_hat).pow(2).sum().item()/2
                #err = (a - a_hat).pow(2).sum().item()*divA + (b - b_hat).pow(2).sum().item()*divB
                #b_hat = torch.matmul(u[:-1], K[:-1,:]) * v#[:-1]
                #err = (b - b_hat).pow(2).sum().item()
            else:   
                b_hat = torch.matmul(u, K) * v
                err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            
            #log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1
    # t2 = time.time()
    # print("Sinkhorn loop: ", t2 - t1)
    #print(it)
    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = reg * torch.log(u + M_EPS)
        log['beta'] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P


def sinkhorn_stabilized(a, b, C, reg=1e-1, maxIter=1000, tau=1e3, stopThr=1e-9,
                        verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1]
    but with the log stabilization proposed in [3] an defined in [2] (Algo 3.1)

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    See Also
    --------

    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        alpha = warm_start['alpha']
        beta = warm_start['beta']
    else:
        alpha = torch.zeros(na, dtype=a.dtype).to(device)
        beta = torch.zeros(nb, dtype=b.dtype).to(device)

    u = torch.ones(na, dtype=a.dtype).to(device) / na
    v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    def update_K(alpha, beta):
        """log space computation"""
        """memory efficient"""
        torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=K)
        torch.add(K, -C, out=K)
        torch.div(K, reg, out=K)
        torch.exp(K, out=K)

    def update_P(alpha, beta, u, v, ab_updated=False):
        """log space P (gamma) computation"""
        torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=P)
        torch.add(P, -C, out=P)
        torch.div(P, reg, out=P)
        if not ab_updated:
            torch.add(P, torch.log(u + M_EPS).reshape(-1, 1), out=P)
            torch.add(P, torch.log(v + M_EPS).reshape(1, -1), out=P)
        torch.exp(P, out=P)

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    update_K(alpha, beta)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1
    ab_updated = False

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)
    P = torch.empty(C.shape, dtype=C.dtype).to(device)

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        ab_updated = False
        # remove numerical problems and store them in K
        if u.abs().sum() > tau or v.abs().sum() > tau:
            alpha += reg * torch.log(u + M_EPS)
            beta += reg * torch.log(v + M_EPS)
            u.fill_(1. / na)
            v.fill_(1. / nb)
            update_K(alpha, beta)
            ab_updated = True

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            update_P(alpha, beta, u, v, ab_updated)
            b_hat = torch.sum(P, 0)
            err = (b - b_hat).pow(2).sum().item()
            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = alpha + reg * torch.log(u + M_EPS)
        log['beta'] = beta + reg * torch.log(v + M_EPS)

    # transport plan
    update_P(alpha, beta, u, v, False)

    if log:
        return P, log
    else:
        return P


def sinkhorn_epsilon_scaling(a, b, C, reg=1e-1, maxIter=100, maxInnerIter=100, tau=1e3, scaling_base=0.75,
                             scaling_coef=None, stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10,
                             print_freq=200, **kwargs):
    """
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [1] but with the log stabilization
    proposed in [3] and the log scaling proposed in [2] algorithm 3.2

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    See Also
    --------

    """

    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    def get_reg(it, reg, pre_reg):
        if it == 1:
            return scaling_coef
        else:
            if (pre_reg - reg) * scaling_base < M_EPS:
                return reg
            else:
                return (pre_reg - reg) * scaling_base + reg

    if scaling_coef is None:
        scaling_coef = C.max() + reg

    it = 1
    err = 1
    running_reg = scaling_coef

    if log:
        log = {'err': []}

    warm_start = None

    while (err > stopThr and it <= maxIter):
        running_reg = get_reg(it, reg, running_reg)
        P, _log = sinkhorn_stabilized(a, b, C, running_reg, maxIter=maxInnerIter, tau=tau,
                                      stopThr=stopThr, verbose=False, log=True,
                                      warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                      **kwargs)

        warm_start = {}
        warm_start['alpha'] = _log['alpha']
        warm_start['beta'] = _log['beta']

        primal_val = (C * P).sum() + reg * (P * torch.log(P)).sum() - reg * P.sum()
        dual_val = (_log['alpha'] * a).sum() + (_log['beta'] * b).sum() - reg * P.sum()
        err = primal_val - dual_val
        log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['alpha'] = _log['alpha']
        log['beta'] = _log['beta']
        return P, log
    else:
        return P