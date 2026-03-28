from LogRegpy.tree.node import Node
import time
from typing import Optional
import math
import cupy as cp
import cupyx.scipy.linalg as cpx_linalg

def cho_solve_gpu(L, b):
    """Solve (L L^T) x = b given lower-triangular Cholesky factor L."""
    y = cpx_linalg.solve_triangular(
        L, b, lower=True, trans='N', overwrite_b=False, check_finite=False
    )
    x = cpx_linalg.solve_triangular(
        L, y, lower=True, trans='T', overwrite_b=False, check_finite=False
    )
    return x

sigmoid_stats = cp.ElementwiseKernel(
    'float32 z, float32 y',
    'float32 p, float32 W, float32 loss, float32 r',
    '''
    p = 1.0 / (1.0 + exp(-z));
    W = p * (1.0 - p);
    loss = -y * log(p + 1e-15) - (1.0 - y) * log(1.0 - p + 1e-15);
    r = p - y;
    ''',
    'sigmoid_stats'
)


def newton_cholesky_logistic_cupy(
    X,
    y,
    w0=None,
    l2_reg=0.0,
    tol=1e-6,
    max_iter=50,
    verbose=False,
):
    """
    Newton–Cholesky solver for binary logistic regression (mean loss) on GPU.
    """

    n, d = X.shape
    if w0 is None:
        w = cp.zeros(d, dtype=X.dtype)
    else:
        w = w0.copy()

    eye = cp.eye(d, dtype=X.dtype)


    def sigmoid(z):
        return 1.0 / (1.0 + cp.exp(-z))

    def loss_and_grad(w):
        # z = X @ w
        # p = sigmoid(z)
        # # ---- mean negative log-likelihood + l2 ----
        # loss = (
        #     -cp.mean(y * cp.log(p + 1e-15) + (1 - y) * cp.log(1 - p + 1e-15))
        #     + 0.5 * l2_reg * cp.dot(w, w)
        # )
        # grad = (X.T @ (p - y)) / n + l2_reg * w
        z = X @ w
        p, W, loss_vec, r = sigmoid_stats(z, y)
        loss = cp.mean(loss_vec) + 0.5 * l2_reg * cp.dot(w, w)
        grad = (X.T @ r) / n + l2_reg * w
        return loss, grad, p, W

    loss, grad, p, W = loss_and_grad(w)

    for it in range(1, max_iter + 1):
        if verbose:
            print(f"Newton iter {it}, loss={loss:.6f}")

        # ---- Hessian (mean loss) ----
        Xw = X * W[:, None]             # (n,d)
        H = (X.T @ Xw) / n + l2_reg * eye

        # ---- Newton step ----
        try:
            L = cp.linalg.cholesky(H)
            delta = cho_solve_gpu(L, -grad)
        except cp.linalg.LinAlgError:
            raise RuntimeError("Hessian not SPD; try increasing l2_reg")

        grad_dot_delta = grad @ delta
        if grad_dot_delta > 0:
            raise RuntimeError("Newton direction is not a descent direction")

        # ---- Backtracking line search (Armijo) ----
        t = 1.0
        beta = 0.5
        sigma = 1e-4


        for _ in range(20):
            w_new = w + t * delta
            loss_new, grad_new, p_new, W_new = loss_and_grad(w_new)

            if loss_new <= loss + sigma * t * grad_dot_delta:
                break
            t *= beta
        else:
            if verbose:
                raise RuntimeError("Line search failed")

        # ---- Update ----
        w = w_new
        loss, grad, p, W = loss_new, grad_new, p_new, W_new

        # ---- Convergence checks ----
        grad_inf = cp.max(cp.abs(grad))

        if grad_inf <= tol:
            dec = 0.5 * (delta @ (H @ delta))
            step_norm = cp.linalg.norm(delta)
            if dec <= tol or step_norm <= tol:
                if verbose:
                    print("Converged (gradient + Newton decrement).")
                break
        elif verbose:
            print(f"     max |gradient| {grad_inf}")

    return w, loss

# def parallel_newton_cholesky(
#     X: cp.ndarray,
#     y: cp.ndarray,
#     nodes: list,
#     lamb: float = 0.0,
#     tol: float = 1e-4,
#     max_iter: int = 50,
#     verbose: bool = False,
# ):
#     """
#     Parallel Newton–Cholesky for logistic regression (mean loss).
#     """

#     start_time = time.time()
#     n, d = X.shape
#     num_models = len(nodes)
#     active = cp.ones(num_models, dtype=cp.bool_)

#     # ---- Build masks and stacked weights ----
#     masks = [
#         cp.array(Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out))
#         for node in nodes
#     ]

#     W_all = cp.zeros((d, num_models), dtype=X.dtype)
#     M = cp.zeros_like(W_all)

#     for i, mask in enumerate(masks):
#         W_all[mask, i] = nodes[i].coefs[mask]
#         M[mask, i] = 1.0

#     eye = cp.eye(d, dtype=X.dtype)

#     # ---- Buffers ----
#     logits = cp.empty((n, num_models), dtype=X.dtype)
#     p = cp.empty_like(logits)
#     W = cp.empty_like(logits)
#     r = cp.empty_like(logits)
#     loss_terms = cp.empty_like(logits)

#     objs = cp.full((num_models,), cp.inf, dtype=X.dtype)

#     # ---- Newton loop ----
#     for it in range(1, max_iter + 1):
#         # ---- Forward pass (batched) ----
#         logits[:] = X @ W_all
#         p[:], W[:], loss_terms[:], r[:] = sigmoid_stats(
#             logits, y[:, None]
#         )

#         # ---- Objective (mean loss) ----
#         loss_vec = cp.sum(loss_terms, axis=0) / n
#         reg = 0.5 * lamb * cp.sum(W_all ** 2, axis=0)
#         new_objs = loss_vec + reg

#         if verbose:
#             print(f"Newton iter {it}, obj mean={cp.mean(new_objs):.6f}")

#         # ---- Gradient (batched) ----
#         grad_all = (X.T @ r) / n + lamb * W_all
#         active = cp.max(cp.abs(grad_all), axis=0) >= tol
#         if verbose:
#             print("grad_inf_all", cp.max(cp.abs(grad_all), axis=0))


#         # ---- Per-model Newton steps ----
#         for j in range(num_models):
#             if not active[j]:
#                 continue

#             grad = grad_all[:, j]
#             Wj = W[:, j]

#             # Hessian
#             Xw = X * Wj[:, None]
#             H = (X.T @ Xw) / n + lamb * eye

#             try:
#                 L = cp.linalg.cholesky(H)
#                 delta = cho_solve_gpu(L, -grad)
#             except cp.linalg.LinAlgError:
#                 raise RuntimeError(f"Hessian not SPD for model {j}")

#             # ---- Line search ----
#             t = 1.0
#             beta = 0.5
#             sigma = 1e-4

#             grad_dot_delta = grad @ delta

#             w_old = W_all[:, j]

#             for _ in range(20):
#                 w_new = w_old + t * delta
#                 z_new = X @ w_new

#                 p_new, _, loss_new_vec, _ = sigmoid_stats(
#                     z_new, y
#                 )
#                 loss_new = (
#                     cp.mean(loss_new_vec)
#                     + 0.5 * lamb * cp.dot(w_new, w_new)
#                 )

#                 if loss_new <= new_objs[j] + sigma * t * grad_dot_delta:
#                     break
#                 t *= beta
#             else:
#                 raise RuntimeError(f"Line search failed for model {j}")

#             # Update
#             W_all[:, j] = w_new

#         # Enforce masks
#         W_all *= M

#         # ---- Convergence ----
#         if not cp.any(active):
#             if verbose:
#                 print("All models converged.")
#             break
#         elif verbose:
#             print(f"iter {it}\n     active: {active}\n    objs: {objs}")

#         objs[:] = new_objs

#     # ---- Write back ----
#     for i, node in enumerate(nodes):
#         node.coefs = W_all[:, i]
#         node.lb = objs[i].item()

#     if verbose:
#         print("Total time:", time.time() - start_time)

#     return nodes

import cupy as cp
import time

def parallel_newton_cholesky(
    X: cp.ndarray,
    y: cp.ndarray,
    nodes: list,
    lamb: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 50,
    verbose: bool = False,
):
    """
    Fully batched Newton–Cholesky for logistic regression (mean loss)
    across multiple models with the same feature count.
    """

    start_time = time.time()
    n, d = X.shape
    num_models = len(nodes)

    # ---- Build masks and initial weights ----
    masks = [
        cp.array(Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out))
        for node in nodes
    ]
    W_all = cp.zeros((d, num_models), dtype=X.dtype)
    M = cp.zeros_like(W_all)
    for i, mask in enumerate(masks):
        W_all[mask, i] = nodes[i].coefs[mask]
        M[mask, i] = 1.0

    eye = cp.eye(d, dtype=X.dtype)

    # ---- Buffers ----
    logits = cp.empty((n, num_models), dtype=X.dtype)
    p = cp.empty_like(logits)
    W_sigmoid = cp.empty_like(logits)
    r = cp.empty_like(logits)
    loss_terms = cp.empty_like(logits)
    objs = cp.full((num_models,), cp.inf, dtype=X.dtype)
    active = cp.ones(num_models, dtype=cp.bool_)

    # ---- Newton loop ----
    for it in range(1, max_iter + 1):
        # ---- Forward pass (batched) ----
        logits[:] = X @ W_all
        p[:], W_sigmoid[:], loss_terms[:], r[:] = sigmoid_stats(logits, y[:, None])

        # ---- Objective (mean loss + L2 reg) ----
        loss_vec = cp.sum(loss_terms, axis=0) / n
        reg = 0.5 * lamb * cp.sum(W_all ** 2, axis=0)
        new_objs = loss_vec + reg

        if verbose:
            print(f"Newton iter {it}, obj mean={cp.mean(new_objs):.6f}")

        # ---- Gradient (batched) ----
        grad_all = (X.T @ r) / n + lamb * W_all
        active = cp.max(cp.abs(grad_all), axis=0) >= tol
        if verbose:
            print("grad_inf_all", cp.max(cp.abs(grad_all), axis=0))
        if not cp.any(active):
            if verbose:
                print("All models converged.")
            break

        # ---- Batched Hessian and Newton step ----
        # Xw: (n, d, num_models)
        X_expanded = X[:, :, None]           # (n, d, 1)
        W_expanded = W_sigmoid[:, None, :]   # (n, 1, num_models)
        Xw = X_expanded * W_expanded
        H_all = (X.T @ Xw.reshape(n, d*num_models)).reshape(d, d, num_models) / n
        H_all += lamb * eye[:, :, None]

        # Batched Cholesky
        L_all = cp.linalg.cholesky(H_all)

        # Batched triangular solve: forward and backward
        delta_all = cp.empty_like(W_all)
        for j in range(num_models):
            if not active[j]:
                continue
            grad = grad_all[:, j]
            L = L_all[:, :, j]
            # Solve L L^T delta = -grad
            y_solve = cp.linalg.solve(L, -grad)
            delta_all[:, j] = cp.linalg.solve(L.T, y_solve)

        # ---- Line search per model ----
        beta = 0.5
        sigma = 1e-4
        for j in range(num_models):
            if not active[j]:
                continue

            w_old = W_all[:, j]
            delta = delta_all[:, j]
            grad_dot_delta = grad_all[:, j] @ delta

            t = 1.0
            for _ in range(20):
                w_new = w_old + t * delta
                z_new = X @ w_new
                p_new, _, loss_new_vec, _ = sigmoid_stats(z_new, y)
                loss_new = cp.mean(loss_new_vec) + 0.5 * lamb * cp.dot(w_new, w_new)
                if loss_new <= new_objs[j] + sigma * t * grad_dot_delta:
                    break
                t *= beta
            else:
                raise RuntimeError(f"Line search failed for model {j}")

            W_all[:, j] = w_new

        # Enforce masks
        W_all *= M
        objs[:] = new_objs

    # ---- Write back results ----
    for i, node in enumerate(nodes):
        node.coefs = W_all[:, i]
        node.lb = objs[i].item()

    if verbose:
        print("Total time:", time.time() - start_time)

    return nodes


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

    print("\n\nsklearn")

    from sklearn.linear_model import LogisticRegression
    for solver in ['newton-cholesky']:
        clf = LogisticRegression(
        penalty=None,
        solver=solver,
        fit_intercept=False,
        max_iter=500,
        verbose=False
        )
        sklearn_time = time.time()
        clf.fit(X, y)
        theta_sklearn = clf.coef_.ravel()
        z = X @ theta_sklearn
        loss = np.logaddexp(0, (1 - 2*y) * z).mean()
        sklearn_time = time.time() - sklearn_time
        print(f"sklearn {solver} time", sklearn_time, "obj", loss)

    print("\n\nCustomCupy")

    for numba in range(3):
        runtime = time.time()
        coef, loss = newton_cholesky_logistic_cupy(X_gpu, y_gpu, verbose=False)
        cg_time = time.time() - runtime   
        print(f"single cholesky {numba} time",cg_time, "cg obj", loss)

    Node.configure(n=X_gpu.shape[1], k=15)

    for numba in range(3):
            new_nodes = []
            for i in range(59):
                new_node = Node(0, 1 << i, coefs = coef.copy())
                new_node.coefs[i] = 0
                new_nodes.append(new_node)
            
            par_time=time.time()
            parallel_newton_cholesky(X_gpu, y_gpu, new_nodes, verbose=True)
            gd_1_objs = np.array([nodd.lb for nodd in new_nodes])
            if numba == 0:
                print("ch1", time.time() - par_time)
                # print(new_nodes[0].coefs)
    
    

    