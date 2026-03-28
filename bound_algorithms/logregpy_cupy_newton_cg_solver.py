


# def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
#                        callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
#                        disp=False, return_all=False, c1=1e-4, c2=0.9,
#                        **unknown_options):
#     """
#     Minimization of scalar function of one or more variables using the
#     Newton-CG algorithm.

#     Note that the `jac` parameter (Jacobian) is required.

#     Options
#     -------
#     disp : bool
#         Set to True to print convergence messages.
#     xtol : float
#         Average relative error in solution `xopt` acceptable for
#         convergence.
#     maxiter : int
#         Maximum number of iterations to perform.
#     eps : float or ndarray
#         If `hessp` is approximated, use this value for the step size.
#     return_all : bool, optional
#         Set to True to return a list of the best solution at each of the
#         iterations.
#     c1 : float, default: 1e-4
#         Parameter for Armijo condition rule.
#     c2 : float, default: 0.9
#         Parameter for curvature condition rule.

#     Notes
#     -----
#     Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
#     """
#     _check_unknown_options(unknown_options)
#     if jac is None:
#         raise ValueError('Jacobian is required for Newton-CG method')
#     fhess_p = hessp
#     fhess = hess
#     avextol = xtol
#     epsilon = eps
#     retall = return_all

#     x0 = asarray(x0).flatten()
#     # TODO: add hessp (callable or FD) to ScalarFunction?
#     sf = _prepare_scalar_function(
#         fun, x0, jac, args=args, epsilon=eps, hess=hess
#     )
#     f = sf.fun
#     fprime = sf.grad
#     _h = sf.hess(x0)

#     # Logic for hess/hessp
#     # - If a callable(hess) is provided, then use that
#     # - If hess is a FD_METHOD, or the output from hess(x) is a LinearOperator
#     #   then create a hessp function using those.
#     # - If hess is None but you have callable(hessp) then use the hessp.
#     # - If hess and hessp are None then approximate hessp using the grad/jac.

#     if (hess in FD_METHODS or isinstance(_h, LinearOperator)):
#         fhess = None

#         def _hessp(x, p, *args):
#             return sf.hess(x).dot(p)

#         fhess_p = _hessp

#     def terminate(warnflag, msg):
#         if disp:
#             _print_success_message_or_warn(warnflag, msg)
#             print("         Current function value: %f" % old_fval)
#             print("         Iterations: %d" % k)
#             print("         Function evaluations: %d" % sf.nfev)
#             print("         Gradient evaluations: %d" % sf.ngev)
#             print("         Hessian evaluations: %d" % hcalls)
#         fval = old_fval
#         result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
#                                 njev=sf.ngev, nhev=hcalls, status=warnflag,
#                                 success=(warnflag == 0), message=msg, x=xk,
#                                 nit=k)
#         if retall:
#             result['allvecs'] = allvecs
#         return result

#     hcalls = 0
#     if maxiter is None:
#         maxiter = len(x0)*200
#     cg_maxiter = 20*len(x0)

#     xtol = len(x0) * avextol
#     update_l1norm = 2 * xtol
#     xk = np.copy(x0)
#     if retall:
#         allvecs = [xk]
#     k = 0
#     gfk = None
#     old_fval = f(x0)
#     old_old_fval = None
#     float64eps = np.finfo(np.float64).eps
#     while update_l1norm > xtol:
#         if k >= maxiter:
#             msg = "Warning: " + _status_message['maxiter']
#             return terminate(1, msg)
#         # Compute a search direction pk by applying the CG method to
#         #  del2 f(xk) p = - grad f(xk) starting from 0.
#         b = -fprime(xk)
#         maggrad = np.linalg.norm(b, ord=1)
#         eta = min(0.5, math.sqrt(maggrad))
#         termcond = eta * maggrad
#         xsupi = zeros(len(x0), dtype=x0.dtype)
#         ri = -b
#         psupi = -ri
#         i = 0
#         dri0 = np.dot(ri, ri)

#         if fhess is not None:             # you want to compute hessian once.
#             A = sf.hess(xk)
#             hcalls += 1

#         for k2 in range(cg_maxiter):
#             if np.add.reduce(np.abs(ri)) <= termcond:
#                 break
#             if fhess is None:
#                 if fhess_p is None:
#                     Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
#                 else:
#                     Ap = fhess_p(xk, psupi, *args)
#                     hcalls += 1
#             else:
#                 # hess was supplied as a callable or hessian update strategy, so
#                 # A is a dense numpy array or sparse matrix
#                 Ap = A.dot(psupi)
#             # check curvature
#             Ap = asarray(Ap).squeeze()  # get rid of matrices...
#             curv = np.dot(psupi, Ap)
#             if 0 <= curv <= 3 * float64eps:
#                 break
#             elif curv < 0:
#                 if (i > 0):
#                     break
#                 else:
#                     # fall back to steepest descent direction
#                     xsupi = dri0 / (-curv) * b
#                     break
#             alphai = dri0 / curv
#             xsupi += alphai * psupi
#             ri += alphai * Ap
#             dri1 = np.dot(ri, ri)
#             betai = dri1 / dri0
#             psupi = -ri + betai * psupi
#             i += 1
#             dri0 = dri1          # update np.dot(ri,ri) for next time.
#         else:
#             # curvature keeps increasing, bail out
#             msg = ("Warning: CG iterations didn't converge. The Hessian is not "
#                    "positive definite.")
#             return terminate(3, msg)

#         pk = xsupi  # search direction is solution to system.
#         gfk = -b    # gradient at xk

#         try:
#             alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
#                      _line_search_wolfe12(f, fprime, xk, pk, gfk,
#                                           old_fval, old_old_fval, c1=c1, c2=c2)
#         except _LineSearchError:
#             # Line search failed to find a better solution.
#             msg = "Warning: " + _status_message['pr_loss']
#             return terminate(2, msg)

#         update = alphak * pk
#         xk += update        # upcast if necessary
#         if retall:
#             allvecs.append(xk)
#         k += 1
#         intermediate_result = OptimizeResult(x=xk, fun=old_fval)
#         if _call_callback_maybe_halt(callback, intermediate_result):
#             return terminate(5, "")
#         update_l1norm = np.linalg.norm(update, ord=1)

#     else:
#         if np.isnan(old_fval) or np.isnan(update).any():
#             return terminate(3, _status_message['nan'])

#         msg = _status_message['success']
#         return terminate(0, msg)








# #SKLEARN
# loss = LinearModelLoss(
#             base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
#         )
# func = loss.loss
# grad = loss.gradient
# hess = loss.gradient_hessian_product
# _newton_cg(
#                 grad_hess=hess,
#                 func=func,
#                 grad=grad,
#                 x0=w0,
#                 args=args,
#                 maxiter=max_iter,
#                 tol=tol,
#                 verbose=verbose,
#             )


# def _newton_cg(
#     grad_hess,
#     func,
#     grad,
#     x0,
#     args=(),
#     tol=1e-4,
#     maxiter=100,
#     maxinner=200,
#     line_search=True,
#     warn=True,
#     verbose=0,
# ):
#     """
#     Minimization of scalar function of one or more variables using the
#     Newton-CG algorithm.

#     Parameters
#     ----------
#     grad_hess : callable
#         Should return the gradient and a callable returning the matvec product
#         of the Hessian.

#     func : callable
#         Should return the value of the function.

#     grad : callable
#         Should return the function value and the gradient. This is used
#         by the linesearch functions.

#     x0 : array of float
#         Initial guess.

#     args : tuple, default=()
#         Arguments passed to func_grad_hess, func and grad.

#     tol : float, default=1e-4
#         Stopping criterion. The iteration will stop when
#         ``max{|g_i | i = 1, ..., n} <= tol``
#         where ``g_i`` is the i-th component of the gradient.

#     maxiter : int, default=100
#         Number of Newton iterations.

#     maxinner : int, default=200
#         Number of CG iterations.

#     line_search : bool, default=True
#         Whether to use a line search or not.

#     warn : bool, default=True
#         Whether to warn when didn't converge.

#     Returns
#     -------
#     xk : ndarray of float
#         Estimated minimum.
#     """
#     x0 = np.asarray(x0).flatten()
#     xk = np.copy(x0)
#     k = 0

#     if line_search:
#         old_fval = func(x0, *args)
#         old_old_fval = None
#     else:
#         old_fval = 0

#     is_verbose = verbose > 0

#     # Outer loop: our Newton iteration
#     while k < maxiter:
#         # Compute a search direction pk by applying the CG method to
#         #  del2 f(xk) p = - fgrad f(xk) starting from 0.
#         fgrad, fhess_p = grad_hess(xk, *args)

#         absgrad = np.abs(fgrad)
#         max_absgrad = np.max(absgrad)
#         check = max_absgrad <= tol
#         if is_verbose:
#             print(f"Newton-CG iter = {k}")
#             print("  Check Convergence")
#             print(f"    max |gradient| <= tol: {max_absgrad} <= {tol} {check}")
#         if check:
#             break

#         maggrad = np.sum(absgrad)
#         eta = min([0.5, np.sqrt(maggrad)])
#         termcond = eta * maggrad

#         # Inner loop: solve the Newton update by conjugate gradient, to
#         # avoid inverting the Hessian
#         xsupi = _cg(fhess_p, fgrad, maxiter=maxinner, tol=termcond, verbose=verbose)

#         alphak = 1.0

#         if line_search:
#             try:
#                 alphak, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
#                     func,
#                     grad,
#                     xk,
#                     xsupi,
#                     fgrad,
#                     old_fval,
#                     old_old_fval,
#                     verbose=verbose,
#                     args=args,
#                 )
#             except _LineSearchError:
#                 warnings.warn("Line Search failed")
#                 break

#         xk += alphak * xsupi  # upcast if necessary
#         k += 1

#     if warn and k >= maxiter:
#         warnings.warn(
#             (
#                 f"newton-cg failed to converge at loss = {old_fval}. Increase the"
#                 " number of iterations."
#             ),
#             ConvergenceWarning,
#         )
#     elif is_verbose:
#         print(f"  Solver did converge at loss = {old_fval}.")
#     return xk, k



#LRP

import cupy as cp


def newton_cg_llogistic_binary(
    X,
    y,
    w0=None,
    tol=1e-4,
    maxiter=100,
    maxinner=200,
    verbose=False,
):
    """
    Newton-CG for binary logistic regression using CuPy.

    Parameters
    ----------
    X : (n_samples, n_features) cupy array
    y : (n_samples,) cupy array with values in {0,1}
    w0 : initial weights (n_features [+1 if intercept])
    fit_intercept : whether to include intercept
    tol : stopping tolerance on infinity norm of gradient
    maxiter : Newton iterations
    maxinner : CG iterations
    """

    n_samples, n_features = X.shape

    if w0 is None:
        w = cp.zeros(n_features)
    else:
        w = w0.copy()

    def sigmoid(z):
        return 1.0 / (1.0 + cp.exp(-z))

    def loss(w):
        z = X @ w
        return cp.mean(cp.log1p(cp.exp(z)) - y * z)

    def grad(w):
        z = X @ w
        p = sigmoid(z)
        return (X.T @ (p - y)) / n_samples

    def hess_vec(v):
        return (X.T @ (W * (X @ v))) / n_samples


    # ----- Conjugate Gradient -----
    def cg(Ax, b, tol, maxiter):
        """
        Solve Ax = -b approximately using CG.
        """
        x = cp.zeros_like(b)
        r = -b - Ax(x)
        p = r.copy()
        rsold = cp.dot(r, r)

        for _ in range(maxiter):
            Ap = Ax(p)
            alpha = rsold / cp.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = cp.dot(r, r)

            if cp.sqrt(rsnew) < tol:
                break

            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x

    # ----- Newton loop -----
    for k in range(maxiter):
        z = X @ w
        p_sig = sigmoid(z)
        W = p_sig * (1 - p_sig)

        g = X.T @ (p_sig - y)
        gnorm = cp.max(cp.abs(g))

        if verbose:
            print(f"Newton iter {k}, ||g||_inf = {gnorm:.3e}")

        if gnorm <= tol:
            break

        # Eisenstat–Walker forcing
        g1 = cp.sum(cp.abs(g))
        eta = min(0.5, cp.sqrt(g1))
        cg_tol = 1e-10 #eta * g1

        # Cached Hessian-vector product
        def hess_vec(v):
            return X.T @ (W * (X @ v))

        p = cg(hess_vec, g, tol=cg_tol, maxiter=maxinner)

        # (line search stays for now)
        alpha = 1.0
        f0 = cp.sum(cp.log1p(cp.exp(z)) - y * z)

        while loss(w + alpha * p) > f0 + 1e-4 * alpha * cp.dot(g, p):
            alpha *= 0.5
            if alpha < 1e-8:
                break

        w += alpha * p


    return w

def newton_cg_logistic_binary(
    X,
    y,
    w0=None,
    tol=1e-4,
    maxiter=50,
    maxinner=50,
    verbose=False,
):
    n_samples, n_features = X.shape

    if w0 is None:
        w = cp.zeros(n_features, dtype=X.dtype)
    else:
        w = w0.copy()

    def sigmoid(z):
        return 1.0 / (1.0 + cp.exp(-z))

    def loss(z):
        return cp.mean(cp.log1p(cp.exp(z)) - y * z)

    for k in range(maxiter):
        z = X @ w
        p_sig = sigmoid(z)
        W = p_sig * (1 - p_sig)

        g = (X.T @ (p_sig - y)) / n_samples
        g_norm = cp.linalg.norm(g)

        if verbose:
            print(f"Newton iter {k}, ||g||_2 = {g_norm:.3e}")

        if g_norm <= tol:
            break

        # ---- sklearn-style forcing term ----
        eta = min(0.5, cp.sqrt(g_norm))
        cg_tol = eta * g_norm

        def hess_vec(v):
            return (X.T @ (W * (X @ v))) / n_samples

        # ---- CG solve ----
        def cg(Ax, b, tol, maxiter):
            x = cp.zeros_like(b)
            r = -b - Ax(x)
            p = r.copy()
            rsold = cp.dot(r, r)

            for _ in range(maxiter):
                Ap = Ax(p)
                alpha = rsold / cp.dot(p, Ap)
                x += alpha * p
                r -= alpha * Ap
                rsnew = cp.dot(r, r)

                if cp.sqrt(rsnew) <= tol:
                    break

                p = r + (rsnew / rsold) * p
                rsold = rsnew

            return x

        p = cg(
            hess_vec,
            g,
            tol=cg_tol,
            maxiter=min(maxinner, n_features),
        )

        # ---- line search (same role as sklearn) ----
        alpha = 1.0
        f0 = loss(z)
        while loss(X @ (w + alpha * p)) > f0 + 1e-4 * alpha * cp.dot(g, p):
            alpha *= 0.5
            if alpha < 1e-8:
                break

        w += alpha * p

    return w



import cupy as cp


def newton_cg_trust_region_logistic_binary(
    X,
    y,
    w0=None,
    tol=1e-4,
    maxiter=100,
    maxinner=200,
    Delta0=1.0,
    Delta_max=100.0,
    verbose=False,
):
    """
    Trust-region Newton-CG for binary logistic regression (CuPy).

    Parameters
    ----------
    X : (n, d) cupy array
    y : (n,) cupy array in {0,1}
    w0 : initial weights (d,)
    tol : infinity-norm gradient tolerance
    maxiter : Newton iterations
    maxinner : CG iterations
    Delta0 : initial trust-region radius
    Delta_max : max trust-region radius
    """

    n_samples, n_features = X.shape

    if w0 is None:
        w = cp.zeros(n_features)
    else:
        w = w0.copy()

    def sigmoid(z):
        return 1.0 / (1.0 + cp.exp(-z))

    def loss_from_z(z):
        return cp.sum(cp.log1p(cp.exp(z)) - y * z)

    # ---- Trust-region helpers ----

    def solve_tau(x, d, Delta):
        # Solve ||x + tau d|| = Delta
        a = cp.dot(d, d)
        b = 2.0 * cp.dot(x, d)
        c = cp.dot(x, x) - Delta * Delta
        disc = b * b - 4 * a * c
        tau = (-b + cp.sqrt(disc)) / (2 * a)
        return tau

    def cg_trust_region(Ax, g, Delta, tol, maxiter):
        """
        Approximately solve H p = -g subject to ||p|| <= Delta
        """
        p = cp.zeros_like(g)
        r = -g
        d = r.copy()
        rTr = cp.dot(r, r)

        for _ in range(maxiter):
            Hd = Ax(d)
            dHd = cp.dot(d, Hd)

            # Negative curvature
            if dHd <= 0:
                tau = solve_tau(p, d, Delta)
                return p + tau * d

            alpha = rTr / dHd
            p_next = p + alpha * d

            # Trust-region boundary
            if cp.linalg.norm(p_next) >= Delta:
                tau = solve_tau(p, d, Delta)
                return p + tau * d

            p = p_next
            r -= alpha * Hd
            rTr_new = cp.dot(r, r)

            if cp.sqrt(rTr_new) <= tol:
                return p

            beta = rTr_new / rTr
            d = r + beta * d
            rTr = rTr_new

        return p

    # ---- Newton loop ----

    Delta = Delta0

    for k in range(maxiter):
        # Cache curvature state
        z = X @ w
        p_sig = sigmoid(z)
        W = p_sig * (1.0 - p_sig)

        g = X.T @ (p_sig - y)
        gnorm = cp.max(cp.abs(g))

        if verbose:
            print(f"Newton iter {k}, ||g||_inf = {gnorm:.3e}, Delta = {Delta:.2e}")

        if gnorm <= tol:
            break

        g1 = cp.sum(cp.abs(g))
        eta = min(0.5, cp.sqrt(g1))
        cg_tol = eta * g1

        def hess_vec(v):
            return X.T @ (W * (X @ v))

        # Solve trust-region subproblem
        p = cg_trust_region(
            hess_vec,
            g,
            Delta=Delta,
            tol=cg_tol,
            maxiter=maxinner,
        )

        # Predicted reduction
        Hp = hess_vec(p)
        pred = -(cp.dot(g, p) + 0.5 * cp.dot(p, Hp))

        # Actual reduction
        z_new = X @ (w + p)
        act = loss_from_z(z) - loss_from_z(z_new)

        rho = act / pred

        # Trust-region update
        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and cp.linalg.norm(p) >= 0.99 * Delta:
            Delta = min(2.0 * Delta, Delta_max)

        # Accept or reject step
        if rho > 0:
            w += p

    return w




if __name__ == "__main__":
    import time
    start_time = time.time()
    print("successfully started")

    from ucimlrepo import fetch_ucirepo 
    import numpy as np
    import pandas as pd
    import json
    from pprint import pprint
    from sklearn.linear_model import LogisticRegression

    print("modules imported:", time.time() - start_time)

    # fetch dataset 
    dataset = fetch_ucirepo(id=579)
    print("successfully collected dataset:", time.time() - start_time)

    # data preprocessing
    X = dataset.data.features.fillna(0).to_numpy()
    y = dataset.data.targets["ZSN"].to_numpy().ravel()
    unique_vals = np.unique(y)
    if len(unique_vals) != 2:
        raise ValueError(f"Expected exactly two unique values, got: {len(unique_vals)}")
    y = (y == unique_vals[1]).astype(int)
    print("dataset cleaned:", time.time() - start_time)

    # move to GPU
    X_gpu = cp.asarray(X, dtype=cp.float32)
    y_gpu = cp.asarray(y, dtype=cp.float32)

    for numba in range(12):
        if numba % 2 == 0:
            runtime = time.time()
            coef = newton_cg_trust_region_logistic_binary(X_gpu, y_gpu, verbose=False)
        elif numba % 2 == 1:
            runtime = time.time()
            coef = newton_cg_logistic_binary(X_gpu, y_gpu, verbose=False)
        z = X_gpu @ coef
        new_loss = cp.mean(cp.logaddexp(0, (1-2*y_gpu) * z))
        cg_time = time.time() - runtime   
        print(f"cg {numba} time",cg_time, "cg obj", new_loss)
    
    

    from sklearn.linear_model import LogisticRegression
    for solver in ["newton-cg", "lbfgs", "newton-cholesky", "sag", "saga"]:
        clf = LogisticRegression(
        penalty=None,
        solver=solver,
        fit_intercept=False,
        max_iter=500,
        )
        sklearn_time = time.time()
        clf.fit(X, y)
        theta_sklearn = clf.coef_.ravel()
        z = X @ theta_sklearn
        loss = np.logaddexp(0, (1 - 2*y) * z).mean()
        sklearn_time = time.time() - sklearn_time
        print(f"sklearn {solver} time", sklearn_time, "obj", loss)




