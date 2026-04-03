print("SHOULD SEE PRINT STATEMENT1")

import cupy as cp
from LogRegpy.tree.node import Node
import time
from typing import Optional
import math


def sigmoid(z):
    return 1.0 / (1.0 + cp.exp(-z))

logistic_loss = cp.ElementwiseKernel(
    'float32 logits, float32 s',
    'float32 out',
    '''
    float margin = s * logits;
    out = margin > 0
        ? margin + log1p(exp(-margin))
        : log1p(exp(margin));
    ''',
    'logistic_loss'
)
sigmoid_residual = cp.ElementwiseKernel(
    'float32 logits, float32 y',
    'float32 out',
    'out = 1.0f / (1.0f + expf(-logits)) - y;',
    'sigmoid_residual'
)

# Fused kernel: compute sigmoid residual and logistic loss in one pass
resid_and_logloss = cp.ElementwiseKernel(
    'float32 logits, float32 y, float32 s',
    'float32 resid, float32 loss_term',
    '''
    // Sigmoid residual
    resid = 1.0f / (1.0f + expf(-logits)) - y;

    // Logistic loss term
    float margin = s * logits;
    loss_term = margin > 0
        ? margin + log1p(expf(-margin))
        : log1p(exp(margin));
    ''',
    'resid_and_logloss'
)

def parallel_gd(X: cp.ndarray,
                     y: cp.ndarray,
                     nodes: list[Node],
                     lamb: float = 0,
                     lr0: float = 0.001,
                     epochs: int = 1500,
                     tol: float = 1e-8,
                     verbose: bool = False):
    """
    GPU-parallel gradient descent for logistic regression.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        nodes: list of Node objects
        lamb: L2 regularization
        lr0: initial learning rate
        epochs: max passes
        tol: relative tolerance for convergence
        verbose: print progress

    Returns:
        nodes: list of Node objects with updated coefs and objective (lb)
    """
    start_time = time.time()
    m, n = X.shape
    num_models = len(nodes)

    # Precompute label signs
    s = (1 - 2 * y)[:, None]  # (m,1) for broadcasting

    # Build masks and warm-start weights
    masks = [cp.array(Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)) for node in nodes]

    # Stack all model weights into one matrix (n_features, num_models)
    W_all_matrix = cp.zeros((n, num_models), dtype=X.dtype)
    for i, mask in enumerate(masks):
        W_all_matrix[mask, i] = nodes[i].coefs[mask]
    
    # Build (n_features, num_models) mask matrix
    M = cp.zeros_like(W_all_matrix)
    for i, mask in enumerate(masks):
        M[mask, i] = 1

    # Convergence trackers
    objs = cp.full((num_models,), cp.inf, dtype=X.dtype)
    new_objs = cp.full((num_models,), cp.inf, dtype=X.dtype)

    for epoch in range(1, epochs + 1):
        lr = lr0 / epoch

        # Compute logits for all models: (m, num_models)
        logits = X @ W_all_matrix

        # Allocate arrays for residuals and loss
        resid = cp.empty_like(logits)
        loss_terms_per_row = cp.empty_like(logits)

        # Fused kernel: compute residuals and logistic loss terms
        resid_and_logloss(logits, y[:, None], s, resid, loss_terms_per_row)

        # Gradient: (n, num_models)
        grad = (X.T @ resid) / m + lamb * W_all_matrix

        # Update weights
        W_all_matrix -= lr * grad
        # Enforce fixed out weights
        W_all_matrix *= M

        # Objective: mean loss + L2 regularization
        loss_terms = cp.sum(loss_terms_per_row, axis=0) / m
        reg_terms = 0.5 * lamb * cp.sum(W_all_matrix**2, axis=0)
        new_objs = loss_terms + reg_terms

        # Convergence check
        rel_change = cp.abs(new_objs - objs) / new_objs
        if cp.all(rel_change < tol):
            if verbose:
                print(f"Converged at epoch {epoch}")
            break

        objs[:] = new_objs
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}, obj mean={cp.mean(objs):.6f}")

    # Final compute objective
    margin = s * (X @ W_all_matrix)
    logistic_terms = cp.where(
        margin > 0,
        margin + cp.log1p(cp.exp(-margin)),
        cp.log1p(cp.exp(margin))
    )
    new_objs = cp.sum(logistic_terms, axis=0) / m + 0.5 * lamb * cp.sum(W_all_matrix**2, axis=0)

    # Write back to nodes
    for i, node in enumerate(nodes):
        node.coefs = W_all_matrix[:, i]
        node.lb = new_objs[i].item()

    if verbose:
        print("Total time:", time.time() - start_time)
    return nodes

def bb_step(W, W_prev, grad, grad_prev, lr_min=1e-8, lr_max=1e5):
    # Project differences onto active subspace
    S = (W - W_prev)
    Y = (grad - grad_prev)

    # Inner products per model
    SS = cp.sum(S * S, axis=0)
    SY = cp.sum(S * Y, axis=0)

    # Avoid division by zero
    lr = SS / cp.maximum(SY, 1e-12)
    # print("BB", SS, SY)

    return cp.clip(lr, lr_min, lr_max)


def parallel_gd2_gpu_kernel(X: cp.ndarray,
                     y: cp.ndarray,
                     nodes: list[Node],
                     lamb: float = 0,
                     lr0: float = 0.001,
                     epochs: int = 1500,
                     tol: float = 1e-4,
                     verbose: bool = False):
    """
    Fully GPU-parallel gradient descent for logistic regression.
    Splits kernel operations for residuals and loss.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        nodes: list of Node objects
        lamb: L2 regularization
        lr0: initial learning rate
        epochs: max passes
        tol: relative tolerance for convergence
        verbose: print progress

    Returns:
        nodes: list of Node objects with updated coefs and objective (lb)
    """
    start_time = time.time()
    m, n = X.shape
    num_models = len(nodes)

    # Precompute label signs
    s = (1 - 2 * y)[:, None]  # (m,1) for broadcasting

    # Build masks and warm-start weights
    masks = [cp.array(Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)) for node in nodes]

    # Stack all model weights into one matrix (n_features, num_models)
    W_all_matrix = cp.zeros((n, num_models), dtype=X.dtype)
    for i, mask in enumerate(masks):
        W_all_matrix[mask, i] = nodes[i].coefs[mask]

    # Build (n_features, num_models) mask matrix
    M = cp.zeros_like(W_all_matrix)
    for i, mask in enumerate(masks):
        M[mask, i] = 1

    # Convergence trackers
    new_objs = cp.full((num_models,), cp.inf, dtype=X.dtype)
    converged = cp.zeros((num_models,), dtype=bool)
    lr = lr0
    for epoch in range(1, epochs + 1):
        # Compute logits for all models: (m, num_models)
        logits = X @ W_all_matrix
        # Residuals for gradient: (m, num_models)
        resid = sigmoid_residual(logits, y[:, None])
        # Gradient: (n, num_models)
        grad = (X.T @ resid) / m + lamb * W_all_matrix
        grad *= M

        # Redefine lr
        if epoch > 1:
            lr = bb_step(W_all_matrix, W_prev, grad, grad_prev)
            # lr = lr0 / epoch

        # Store previous
        W_prev = W_all_matrix.copy()
        grad_prev = grad.copy()

        # Update weights
        W_all_matrix -= lr * grad
        # W_all_matrix *= M
        
        # Convergence check
        grad_norm = cp.max(cp.abs(grad), axis=0)
        converged |= (grad_norm <= tol)

        if epoch % 10 == 0 and verbose:
            print(f"Epoch {epoch}, converged models={cp.sum(converged)}, lr = {lr}, ||g|| = {grad_norm}")
        if epoch % 10 == 0 and cp.all(converged):
            if verbose:
                print(f"All models converged at epoch {epoch}")
            break

    # Compute objective
    logits = X @ W_all_matrix
    logistic_terms = logistic_loss(logits, s)
    # assert cp.all(logistic_terms >= 0), "Loss has negative terms!"
    new_objs = cp.sum(logistic_terms, axis=0) / m + 0.5 * lamb * cp.sum(W_all_matrix**2, axis=0)
    # print("WENT TO EPOCH", epoch)
    # print("HERE ARE THE NEW_OBJS", new_objs)
    # print(cp.count_nonzero(W_all_matrix, axis=0))

    # Write back to nodes
    for i, node in enumerate(nodes):
        node.coefs = W_all_matrix[:, i]
        node.lb = new_objs[i].item()

    if verbose:
        print("Total time:", time.time() - start_time)
    return cp.sum(converged)

def parallel_nesterov_gd(X: cp.ndarray,
                     y: cp.ndarray,
                     nodes: list[Node],
                     lamb: float = 0,
                     lr0: float = 0.001,
                     beta: float = 0.9,
                     epochs: int = 1500,
                     tol: float = 1e-4,
                     verbose: bool = False):
    """
    Fully GPU-parallel Nesterov GD for logistic regression.
    Splits kernel operations for residuals and loss.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        nodes: list of Node objects
        lamb: L2 regularization
        lr0: initial learning rate
        epochs: max passes
        tol: relative tolerance for convergence
        verbose: print progress

    Returns:
        nodes: list of Node objects with updated coefs and objective (lb)
    """
    start_time = time.time()
    m, n = X.shape
    num_models = len(nodes)

    # Precompute label signs
    s = (1 - 2 * y)[:, None]  # (m,1) for broadcasting

    # Build masks and warm-start weights
    masks = [cp.array(Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)) for node in nodes]

    # Stack all model weights into one matrix (n_features, num_models)
    W_all_matrix = cp.zeros((n, num_models), dtype=X.dtype)
    for i, mask in enumerate(masks):
        W_all_matrix[mask, i] = nodes[i].coefs[mask]

    # Build (n_features, num_models) mask matrix
    M = cp.zeros_like(W_all_matrix, dtype=X.dtype)
    for i, mask in enumerate(masks):
        M[mask, i] = 1

    # velocity initialization
    V = cp.zeros_like(W_all_matrix, dtype=X.dtype)

    # Type enforcement
    beta = cp.array(beta, dtype=cp.float32)

    # Convergence trackers
    converged = cp.zeros((num_models,), dtype=bool)

    # Compute initial objective
    logits = X @ W_all_matrix
    logistic_terms = logistic_loss(logits, s)
    # assert cp.all(logistic_terms >= 0), "Loss has negative terms!"
    new_objs = cp.sum(logistic_terms, axis=0) / m + 0.5 * lamb * cp.sum(W_all_matrix**2, axis=0)
    print("HERE ARE THE intial objectives", new_objs)

    for epoch in range(1, epochs + 1):
        lr = lr0 # / (epoch ** (1/5))  For whatever schedule used

        # Lookahead point
        Y = W_all_matrix + beta * V
        Y *= M   # enforce masking

        # Forward pass
        logits = cp.clip(X @ Y, -20, 20)
        resid = sigmoid_residual(logits, y[:, None])

        # Gradient at lookahead
        grad = (X.T @ resid) / m + lamb * Y
        grad *= M

        # === NESTEROV: velocity + parameter update ===
        V = beta * V - lr * grad
        W_all_matrix += V
        W_all_matrix *= M

        logits_W = cp.clip(X @ W_all_matrix,-20,20)
        resid_W = sigmoid_residual(logits_W, y[:, None])
        grad_W = (X.T @ resid_W) / m + lamb * W_all_matrix
        grad_W *= M
        
        # Convergence check
        grad_norm_inf = cp.max(cp.abs(V), axis=0)
        converged |= (grad_norm_inf <= tol)

        if verbose:
            print(f"Epoch {epoch}, converged models={cp.sum(converged)}, lr = {lr}, ||g||_inf = {grad_norm_inf}")
        if cp.all(converged):
            if verbose:
                print(f"All models converged at epoch {epoch}")
            break

    # Compute objective
    logits = X @ W_all_matrix
    logistic_terms = logistic_loss(logits, s)
    # assert cp.all(logistic_terms >= 0), "Loss has negative terms!"
    new_objs = cp.sum(logistic_terms, axis=0) / m + 0.5 * lamb * cp.sum(W_all_matrix**2, axis=0)
    print("HERE ARE THE NEW_OBJS", new_objs)

    # Write back to nodes
    for i, node in enumerate(nodes):
        node.coefs = W_all_matrix[:, i]
        node.lb = new_objs[i].item()

    if verbose:
        print("Total time:", time.time() - start_time)
    return nodes


# def parallel_dual_gd(X: cp.ndarray,
#                      y: cp.ndarray,
#                      nodes: list[Node],
#                      lamb: float,
#                      lr0: float = 0.001,
#                      epochs: int = 1500,
#                      tol: float = 1e-4,
#                      verbose: bool = False):
#     """
#     Parallel dual gradient ascent with logistic reparameterization and primal recovery for L2-regularized logistic regression.

#     Args:
#         X: (m,n) CuPy array, feature matrix
#         y: (m,) CuPy array, binary labels {0,1}
#         nodes: list[Node], branch-and-bound nodes
#         lamb: L2 regularization parameter
#         lr0: initial learning rate
#         epochs: maximum number of iterations
#         tol: relative convergence tolerance on dual objective
#         verbose: print convergence info

#     Returns:
#         nodes: updated nodes with primal coefs and lower bounds
#     """
#     start_time = time.time()
#     m, n = X.shape

#     # Precompute label signs
#     s = (1 - 2 * y)  # shape (m,)

#     # Build per-node feature masks
#     masks = [
#         cp.array(Node.varbitset_to_list(
#             Node.universal_varbitset & ~node.fixed_out
#         ))
#         for node in nodes
#     ]

#     # Initialize dual variables
#     alphas = [sigmoid(-s * (X[masks[i]] @ node.coefs)) for i,node in enumerate(nodes)]

#     objs = [-cp.inf for _ in nodes]
#     new_objs = [-cp.inf for _ in nodes]
#     converged = [False for _ in nodes]

#     for epoch in range(1, epochs + 1):
#         lr = lr0 / epoch

#         for i, node in enumerate(nodes):
#             if converged[i]:
#                 continue

#             alpha = alphas[i]
#             mask = masks[i]
#             X_sub = X[mask]

#             # Aggregate primal vector v = (1/m) sum s_i alpha_i X_i
#             v = (X.T @ (s * alpha)) / m

#             # Dual gradient
#             logits = X @ v
#             grad = (
#                 -(1 / m) * cp.log(alpha / (1 - alpha))
#                 - (1 / (lamb * m)) * s * logits
#             )

#             # Logistic reparameterized ascent
#             logit_alpha = cp.log(alpha / (1 - alpha))
#             logit_alpha += lr * grad
#             alpha_new = sigmoid(logit_alpha)
#             node.alpha = alpha_new

#             # Dual objective (lower bound)
#             entropy = alpha_new * cp.log(alpha_new) + (1 - alpha_new) * cp.log(1 - alpha_new)
#             quad = cp.sum(v[mask] ** 2)
#             new_objs[i] = (
#                 -(1 / m) * cp.sum(entropy)
#                 - quad / (2 * lamb)
#             )

#             # Convergence check
#             if cp.abs(new_objs[i] - objs[i]) / (cp.abs(new_objs[i]) + 1e-12) < tol:
#                 converged[i] = True
#             objs[i] = new_objs[i]

#         if verbose:
#             print(f"Epoch {epoch}, converged={sum(converged)}")

#         if all(converged) and epoch > 5:
#             if verbose:
#                 print(f"All nodes converged at epoch {epoch}")
#             break

#     for i,node in enumerate(nodes):
#         # Aggregate primal vector v = (1/m) sum s_i alpha_i X_i
#         v = (X.T @ (s * alpha)) / m

#         # Primal recovery (restricted to active features)
#         theta = cp.zeros(n)
#         theta[mask] = -v[mask] / lamb
#         node.coefs = theta

#     # Write lower bounds back to nodes
#     for i, node in enumerate(nodes):
#         node.lb = new_objs[i]

#     return nodes

def parallel_dual_gd_gpu_bb(
    X: cp.ndarray,
    y: cp.ndarray,
    nodes: list[Node],
    lamb: float,
    lr0: float = 1e-2,
    epochs: int = 1000,
    tol: float = 1e-4,
    verbose: bool = False
):
    """
    Fully GPU-parallel dual gradient ascent with
    - logistic reparameterization
    - per-model Barzilai–Borwein steps in Z-space
    - correct dual objective (constant restored)
    - consistent primal/dual geometry
    """

    start_time = time.time()
    m, n = X.shape
    num_models = len(nodes)

    # Label signs
    s = (1 - 2 * y)[:, None]          # (m,1)

    # Feature masks (used ONLY in primal recovery)
    masks = [
        cp.array(Node.varbitset_to_list(
            Node.universal_varbitset & ~node.fixed_out
        ))
        for node in nodes
    ]

    # Build mask matrix for recovery
    M = cp.zeros((n, num_models), dtype=X.dtype)
    for j, mask in enumerate(masks):
        M[mask, j] = 1.0

    # Build warm-start theta matrix
    Theta_all = cp.zeros((n, num_models), dtype=X.dtype)
    for j, mask in enumerate(masks):
        Theta_all[mask, j] = nodes[j].coefs[mask]

    # Warm-start dual from KKT
    logits = X @ Theta_all
    Alpha = sigmoid(-s * logits)
    Alpha = cp.clip(Alpha, 1e-6, 1 - 1e-6)   # numerical safety
    Z = cp.log(Alpha / (1 - Alpha))


    # BB storage (Z-space!)
    Z_prev = None
    Gz_prev = None
    lr = cp.full(num_models, lr0, dtype=X.dtype)

    # Convergence
    converged = cp.zeros(num_models, dtype=bool)

    for epoch in range(1, epochs + 1):

        # v = (1/m) X^T (s ⊙ α)
        V_all = (X.T @ (s * Alpha)) / m        # (n, num_models)

        # logits = X v
        logits = X @ V_all                    # (m, num_models)

        # Dual gradient wrt α
        Grad_alpha = (
            -(1.0 / m) * cp.log(Alpha / (1.0 - Alpha))
            - (1.0 / (lamb * m)) * s * logits
        )

        # Chain rule: gradient wrt Z
        Gz = Grad_alpha * Alpha * (1.0 - Alpha)

        # Per-model BB step in Z-space
        if epoch > 1:
            S = Z - Z_prev
            Y = Gz - Gz_prev

            SS = cp.sum(S * S, axis=0)
            SY = cp.sum(S * Y, axis=0)

            lr = SS / (SY + 1e-12)
            lr = cp.clip(lr, 1e-6, 1.0)

        # Store previous
        Z_prev = Z.copy()
        Gz_prev = Gz.copy()

        # Z update
        Z += lr[None, :] * Gz

        # Back to α
        Alpha = sigmoid(Z)

        # Convergence: per-model infinity norm
        grad_inf = cp.max(cp.abs(Gz), axis=0)
        converged |= (grad_inf < tol)

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch}, "
                f"converged={int(cp.sum(converged))}/{num_models}, "
                f"lr=[{cp.min(lr):.2e}, {cp.max(lr):.2e}]"
            )

        if epoch % 10 == 0 and cp.all(converged):
            break

    # ---------- Dual objective (CORRECTLY SHIFTED) ----------

    entropy = (
        Alpha * cp.log(Alpha)
        + (1.0 - Alpha) * cp.log(1.0 - Alpha)
    )

    quad = cp.sum(V_all * V_all, axis=0)   # NO masking here

    dual = (
        -(1.0 / m) * cp.sum(entropy, axis=0)
        - quad / (2.0 * lamb)
        # + cp.log(2.0)                      # restore constant
    )

    # ---------- Primal recovery ----------

    Theta_all = -(1.0 / lamb) * V_all
    Theta_all *= M                         # masking ONLY here

    # Write back to nodes
    for j, node in enumerate(nodes):
        node.coefs = Theta_all[:, j]
        node.lb = dual[j].item()

    if verbose:
        print("Final duals:", dual)
        print("Total time:", time.time() - start_time)

    return nodes


def single_gd(X: cp.ndarray,
              y: cp.ndarray,
              lamb: float = 0.005,
              warm_start_coefs: Optional[cp.ndarray] = None,
              lr0: float = 0.001,
              epochs: int = 1500,
              tol: float = 1e-4,
              verbose: bool = False,
              ):
    """
    Single GD for logistic regression with early stopping based on log loss convergence.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        warm_start_coefs: (n,) CuPy array
        lr0: initial learning rate
        epochs: maximum number of passes through data
        tol: tolerance for convergence (stopping criterion)
        verbose: if True, prints convergence info

    Returns:
        coefs: list[float], updated coefs
        new_loss: float
        elapsed_time: float, seconds
    """
    start_time = time.time()
    if warm_start_coefs is not None:
        coefs = warm_start_coefs
    else:
        coefs = cp.zeros_like(X[0]) # array with shape equal to no. of features
    
    m = cp.shape(X)[0]
    s = 1 - 2 * y

    # Performing Gradient Descent Optimization for every epoch
    loss = cp.inf
    for epoch in range(1,epochs+1):
        lr = lr0# / math.sqrt(epoch)
        z = X @ coefs     # shape (n,)
        pred = sigmoid(z)
        grad = (X.T @ (pred - y)) / m + lamb * coefs

        # Redefine lr
        if epoch > 1:
            lr = bb_step(coefs, coefs_prev, grad, grad_prev)
            # lr = lr0 / epoch

        # Store previous
        coefs_prev = coefs.copy()
        grad_prev = grad.copy()

        coefs -= lr * grad
        # Recompute loss at updated weights
        # grad_norm = cp.linalg.norm(grad)
        if epoch % 10 == 0 and cp.max(cp.abs(grad), axis=0) < tol:
            # if verbose:
            #     print("Converged at epoch", epoch, "grad_norm_inf", grad_norm_inf)
            break
        # elif epoch % 1 == 0 and verbose:
        #     print("epoch", epoch, "grad norm_inf", grad_norm_inf, "lr", lr)
    z = X @ coefs
    margin = (1 - 2*y) * z
    logistic_terms = cp.where(
        margin > 0,
        margin + cp.log1p(cp.exp(-margin)),
        cp.log1p(cp.exp(margin))
    )
    new_loss = cp.mean(logistic_terms) + 0.5 * lamb * cp.sum(coefs**2)
    return coefs, new_loss


def single_nesterov_gd(X: cp.ndarray,
              y: cp.ndarray,
              lamb: float = 0.005,
              warm_start_coefs: Optional[cp.ndarray] = None,
              lr0: Optional[float] = 0.001,
              beta: float = 0.9,
              epochs: int = 1500,
              tol: float = 1e-4,
              verbose: bool = False,
              ):
    """
    Single GD for logistic regression with early stopping based on log loss convergence.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        warm_start_coefs: (n,) CuPy array
        lr0: initial learning rate
        epochs: maximum number of passes through data
        tol: tolerance for convergence (stopping criterion)
        verbose: if True, prints convergence info

    Returns:
        coefs: list[float], updated coefs
        new_loss: float
        elapsed_time: float, seconds
    """
    start_time = time.time()
    if warm_start_coefs is not None:
        coefs = warm_start_coefs
    else:
        coefs = cp.zeros_like(X[0]) # array with shape equal to no. of features
    
    m = cp.shape(X)[0]
    s = 1 - 2 * y

    # velocity initialization
    V = cp.zeros_like(coefs, dtype=X.dtype)

    # Type enforcement
    beta = cp.array(beta, dtype=cp.float32)

    # Performing Gradient Descent Optimization for every epoch
    loss = cp.inf
    for epoch in range(1,epochs+1):
        lr = lr0# / math.sqrt(epoch)
        
        # Lookahead point
        Y = coefs + beta * V

        # Forward pass
        logits = cp.clip(X @ Y, -20, 20)
        resid = sigmoid_residual(logits, y)

        # Gradient at lookahead
        grad = (X.T @ resid) / m + lamb * Y

        # === NESTEROV: velocity + parameter update ===
        V = beta * V - lr * grad
        coefs += V

        logits_W = cp.clip(X @ coefs,-20,20)
        resid_W = sigmoid_residual(logits_W, y)
        grad_W = (X.T @ resid_W) / m + lamb * coefs
        
        # Convergence check
        grad_norm_inf = cp.max(cp.abs(grad_W))

        if verbose:
            print(f"Epoch {epoch}, converged={grad_norm_inf <= tol}, lr = {lr}, ||g||_inf = {grad_norm_inf}")
        if grad_norm_inf <= tol:
            if verbose:
                print(f"All models converged at epoch {epoch}")
            break

    z = X @ coefs
    margin = (1 - 2*y) * z
    logistic_terms = cp.where(
        margin > 0,
        margin + cp.log1p(cp.exp(-margin)),
        cp.log1p(cp.exp(margin))
    )
    new_loss = cp.mean(logistic_terms) + 0.5 * lamb * cp.sum(coefs**2)
    return coefs, new_loss


def single_kernel_gd(X: cp.ndarray,
              y: cp.ndarray,
              lamb: Optional[float] = 0.005,
              warm_start_coefs: Optional[cp.ndarray] = None,
              lr0: Optional[float] = 0.001,
              epochs: Optional[int] = 1500,
              tol: Optional[float] = 1e-10,
              verbose: Optional[bool] = False,
              ):
    """
    Single GD for logistic regression with early stopping based on log loss convergence.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        warm_start_coefs: (n,) CuPy array
        lr0: initial learning rate
        epochs: maximum number of passes through data
        tol: tolerance for convergence (stopping criterion)
        verbose: if True, prints convergence info

    Returns:
        coefs: list[float], updated coefs
        new_loss: float
        elapsed_time: float, seconds
    """
    start_time = time.time()
    if warm_start_coefs is not None:
        coefs = warm_start_coefs
    else:
        coefs = cp.zeros_like(X[0]) # array with shape equal to no. of features
    
    m = cp.shape(X)[0]
    s = 1 - 2 * y

    # Performing Gradient Descent Optimization for every epoch
    loss = cp.inf
    for epoch in range(1,epochs+1):
        lr = lr0 / epoch
        logits = X @ coefs
        resid = sigmoid_residual(logits, y)
        grad = (X.T @ resid) / m + lamb * coefs
        coefs -= lr * grad
        # Recompute loss at updated weights
        logits = X @ W_all_matrix
        logistic_terms = logistic_loss(logits, s)
        new_loss = cp.mean(logistic_terms) + 0.5 * lamb * cp.sum(coefs**2)
        if cp.abs(new_loss - loss)/new_loss < tol:
            if verbose:
                print("Converged at epoch", epoch, "loss", new_loss)
            break
        elif verbose:
            print("epoch", epoch, "relative improvement", cp.abs(new_loss - loss)/new_loss, "loss", new_loss, "lr", LR)
        loss = new_loss
    return coefs, new_loss


def single_dual_gd(X: cp.ndarray,
              y: cp.ndarray,
              lamb: Optional[float] = 0.005,
              warm_start_coefs: Optional[cp.ndarray] = None,
              lr0: Optional[float] = 4e-4,
              epochs: Optional[int] = 1500,
              tol: Optional[float] = 1e-15,
              verbose: Optional[bool] = False,
              ):
    """
    Single Dual GD for logistic regression with early stopping based on log loss convergence.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        warm_start_coefs: (n,) CuPy array
        lr0: initial learning rate
        epochs: maximum number of passes through data
        tol: tolerance for convergence (stopping criterion)
        verbose: if True, prints convergence info

    Returns:
        coefs: list[float], updated coefs
        new_loss: float
        elapsed_time: float, seconds
    """
    start_time = time.time()
    if warm_start_coefs is not None:
        coefs = warm_start_coefs
    else:
        coefs = cp.zeros_like(X[0]) # array with shape equal to no. of features
    
    m, n = X.shape

    print(coefs)
    # Precompute label signs
    s = (1 - 2 * y)  # shape (m,)

    # Initialize dual variables
    alpha = sigmoid(-s * (X @ coefs))

    # Aggregate primal vector v = (1/m) sum s_i alpha_i X_i
    v = (X.T @ (s * alpha)) / m
    # Dual objective (lower bound)
    entropy = alpha * cp.log(alpha) + (1 - alpha) * cp.log(1 - alpha)
    quad = cp.sum(v ** 2)
    print("original obj",
        -(1 / m) * cp.sum(entropy)
        - quad / (2 * lamb)
    )

    # Performing Gradient Descent Optimization for every epoch
    obj = cp.inf
    lr = lr0
    for epoch in range(1,epochs+1):
        # lr = lr0 / epoch
        # Aggregate primal vector v = (1/m) sum s_i alpha_i X_i
        v = (X.T @ (s * alpha)) / m
        # Dual gradient
        logits = X @ v
        grad = (
            -(1 / m) * cp.log(alpha / (1 - alpha))
            - (1 / (lamb * m)) * s * logits
        )
        # Logistic reparameterized ascent
        logit_alpha = cp.log(alpha / (1 - alpha))
        logit_alpha += lr * grad
        alpha = sigmoid(logit_alpha)
        # Dual objective (lower bound)
        entropy = alpha * cp.log(alpha) + (1 - alpha) * cp.log(1 - alpha)
        quad = cp.sum(v ** 2)
        new_obj = (
            -(1 / m) * cp.sum(entropy)
            - quad / (2 * lamb)
        )
        if cp.abs(new_obj - obj)/(cp.abs(new_obj) + 1e-12) < tol:
            if verbose:
                print("Converged at epoch", epoch, "obj", new_obj)
            break
        elif verbose:
            print("epoch", epoch, "relative improvement", cp.abs(new_obj - obj)/new_obj, "obj", new_obj, "lr", lr)
        obj = new_obj

        # Aggregate primal vector v = (1/m) sum s_i alpha_i X_i
        v = (X.T @ (s * alpha)) / m

        # Primal recovery
        coefs = -v / lamb
    
    return coefs, new_obj


def single_sgd(X: cp.ndarray,
              y: cp.ndarray,
              warm_start_coefs: Optional[cp.ndarray] = None,
              lr0: Optional[float] = 0.9,
              epochs: Optional[int] = 1500,
              tol: Optional[float] = 1e-8,
              verbose: Optional[bool] = False):
    """
    Single SGD for logistic regression with early stopping based on log loss convergence.

    Args:
        X: (m,n) CuPy array, feature matrix
        y: (m,) CuPy array, binary labels {0,1}
        lr0: initial learning rate
        epochs: maximum number of passes through data
        tol: tolerance for convergence (stopping criterion)
        verbose: if True, prints convergence info

    Returns:
        nodes: list[Node], updated nodes
        elapsed_time: float, seconds
    """
    start_time = time.time()
    if warm_start_coefs is not None:
        coefs = warm_start_coefs
    else:
        coefs = cp.zeros_like(X[0]) # array with shape equal to no. of features
    
    # Performing Stochastic Gradient Descent Optimization
    # for every epoch
    loss = cp.inf
    # print("SPLIT", epocha)
    for epoch in range(1,epochs+1):
        lr = lr0 / (epoch)
        # for every data point(X_train,y_train)
        for i in range(len(X)):
            gr_wrt_i = X[i] * (sigmoid(cp.dot(coefs.T, X[i])) - y[i])
            coefs -= lr * gr_wrt_i
        # Recompute loss at updated weights
        z = X @ coefs
        margin = (1 - 2*y) * z
        logistic_terms = cp.where(
            margin > 0,
            margin + cp.log1p(cp.exp(-margin)),
            cp.log1p(cp.exp(margin))
        )
        new_loss = cp.sum(logistic_terms)
        if cp.abs(new_loss - loss)/new_loss < tol and epoch > 5:
            break
        loss = new_loss
    return coefs, new_loss

import random

def forward_stepwise(X_gpu, y_gpu, n, goal_varbitset, lambda_):
    added_varbitset = 0


    coef = cp.zeros(n, dtype=cp.float32)

    for z in range(n):
        new_nodes = []
        vars_to_check = Node.varbitset_to_list(Node.universal_varbitset & ~added_varbitset)
        random.shuffle(vars_to_check)
        for i in vars_to_check:
            new_node = Node(0, Node.universal_varbitset & ~ (added_varbitset | (1 << i)), coefs = coef.copy())
            new_nodes.append(new_node)
        converged = 0
        converged = parallel_gd2_gpu_kernel(X_gpu, y_gpu, new_nodes, lambda_, verbose=False, tol=1e-4, epochs=10000)
        if converged < len(new_nodes):
            print("Convergence check:", converged, "/", len(new_nodes))
        next_var = -1
        worst_obj = math.inf
        for i,j in enumerate(vars_to_check):
            if new_nodes[i].lb < worst_obj:
                next_var = j
                worst_obj = new_nodes[i].lb
                coef = new_nodes[i].coefs
        added_varbitset |= Node.var_to_varbitset(next_var)
        print(z,goal_varbitset == (added_varbitset & goal_varbitset), (added_varbitset & goal_varbitset).bit_count())
        if goal_varbitset == (added_varbitset & goal_varbitset):
            break

import numpy as np
from scipy.special import expit

def generate_logistic_data(
    N,
    n,
    m,
    rho=0.3,
    beta_scale=1.0,
    corr_type="ar1",
    seed=None,
    rng=None
):
    """
    Generate logistic regression data with known sparse support.

    Parameters
    ----------
    N : int
        Number of samples
    n : int
        Number of features
    m : int
        Sparsity level (true number of active variables)
    rho : float
        Correlation parameter (0 <= rho < 1)
    beta_scale : float
        Magnitude of nonzero coefficients
    corr_type : str
        'ar1' or 'block'
    seed : int or None

    Returns
    -------
    X : (N, n) ndarray
    y : (N,) ndarray
    beta_true : (n,) ndarray
    support : ndarray of indices
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # --- True support ---
    support = np.arange(m)  # first m features are active

    # --- True coefficients ---
    beta_true = np.zeros(n)
    signs = rng.choice([-1.0, 1.0], size=m)
    beta_true[support] = beta_scale * signs

    # --- Covariance matrix ---
    if corr_type == "ar1":
        idx = np.arange(n)
        Sigma = rho ** np.abs(np.subtract.outer(idx, idx))

    elif corr_type == "block":
        block_size = 10
        Sigma = np.eye(n)
        for i in range(0, n, block_size):
            block = slice(i, min(i + block_size, n))
            Sigma[block, block] = rho
            np.fill_diagonal(Sigma[block, block], 1.0)

    else:
        raise ValueError("Unknown corr_type")

    # --- Sample X ---
    X = rng.multivariate_normal(mean=np.zeros(n), cov=Sigma, size=N)

    # --- Linear predictor ---
    logits = X @ beta_true

    # --- Binary labels ---
    probs = expit(logits)
    y = rng.binomial(1, probs)

    return X, y, beta_true, support


if __name__ == '__main__':
    configs = {
    "n": [50, 250, 1000],
    "m": [3, 5, 10],
    "N": [500, 1500],
    "rho": [0.0, 0.3, 0.6, 0.9],
    "beta_scale": [0.5, 1.0, 2.0]
    }
    from LogRegpy.tree.node import Node
                
    for n in configs["n"]:
        for m in configs["m"]:
            for N in configs["N"]:
                for run in range(30):
                    print("Testing:",n,m,N, run)
                    X,y,beta_true, support = generate_logistic_data(N,n,m, rho=0.85, beta_scale=0.5)
                    Node.configure(n=n, k=n)
                    support = [int(j) for j in support]
                    support_varbitset = Node.iter_to_varbitset(support)
                    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                    X_gpu = cp.asarray(X, dtype=cp.float32)
                    y_gpu = cp.asarray(y, dtype=cp.float32)
                    forward_stepwise(X_gpu, y_gpu, n, support_varbitset, 0)
                


if __name__ == "__main__" and False:
    print("SHOULD SEE PRINT STATEMENT")
    import time
    start_time = time.time()
    print("successfully started")

    from ucimlrepo import fetch_ucirepo 
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    print("modules imported:", time.time() - start_time)

    # # fetch dataset 
    # dataset = fetch_ucirepo(id=579)
    # print("successfully collected dataset:", time.time() - start_time)

    # # data preprocessing
    # X = dataset.data.features.fillna(0).to_numpy()
    # y = dataset.data.targets["ZSN"].to_numpy().ravel()
    # unique_vals = np.unique(y)
    # if len(unique_vals) != 2:
    #     raise ValueError(f"Expected exactly two unique values, got: {len(unique_vals)}")
    # y = (y == unique_vals[1]).astype(int)

    X = np.loadtxt("LogRegpy/tests/datasets/madelon/MADELON/madelon_train.data")
    y = np.loadtxt("LogRegpy/tests/datasets/madelon/MADELON/madelon_train.labels").ravel()
    unique_vals = np.unique(y)
    if len(unique_vals) != 2:
        raise ValueError(f"Expected exactly two unique values, got: {len(unique_vals)}")
    y = (y == unique_vals[1]).astype(int)

    m,n = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    print("dataset cleaned:", time.time() - start_time)

    # move to GPU
    X_gpu = cp.asarray(X, dtype=cp.float32)
    y_gpu = cp.asarray(y, dtype=cp.float32)

    print("SHAPE", X_gpu.shape)
    
    # Parameters
    tolle = 1e-4
    print("Tolerance", tolle)
    lambda_ = 0.1/m
    print("regularization strength:", lambda_)
    runtime = time.time()
    # sigma_max = cp.linalg.svd(X_gpu, compute_uv=False)[0]
    # L = 0.25 * sigma_max**2 + lambda_
    # lr = 5.0 / L
    # beta = (cp.sqrt(L) - cp.sqrt(lambda_)) / (cp.sqrt(L) + cp.sqrt(lambda_))
    # print("Theoretical lr:", lr, "theoretical beta:", beta, "time:", time.time()-runtime)

    # Solving initial model with Cupy
    for numba in range(3):
        cupy_time = time.time()
        cupy_coefs, cupy_loss = single_gd(X_gpu, y_gpu, lamb=lambda_,  verbose=True, tol=tolle, epochs=5000)
        if numba == 2:
            print("cupy time:", time.time() - cupy_time, "cupy obj:", cupy_loss)

    # Solving initial model with sklearn 
    if lambda_ >0:
        clf = LogisticRegression(
        penalty="l2",
        C=1/(lambda_ * m),
        solver="newton-cg",
        fit_intercept=False,
        tol = tolle
        )
    else:
        clf = LogisticRegression(
        penalty=None,
        solver="newton-cg",
        fit_intercept=False,
        tol = tolle
        )
    sklearn_time = time.time()
    clf.fit(X, y)
    theta_sklearn = clf.coef_.ravel()
    # print("sklearntime", time.time() - sklearn_time)
    # z = X @ theta_sklearn
    # loss = np.logaddexp(0, (1 - 2*y) * z).mean()
    # reg = 0.5 * lambda_ * np.dot(theta_sklearn, theta_sklearn)
    # print("sklearn obj", loss + reg)

    # # Solving 1 feature out model with Cupy
    # cupy_time = time.time()
    # cupy_coefs, cupy_loss = single_gd(X_gpu[:,list(range(1,n))], y_gpu, warm_start_coefs=cp.asarray(theta_sklearn[list(range(1,n))], dtype=cp.float32), lamb=lambda_, lr0=lr,  verbose=False, tol=tolle, epochs=1500)
    # print("cupy time:", time.time() - cupy_time, "cupy obj:", cupy_loss)

    from LogRegpy.tree.node import Node
    Node.configure(n=X_gpu.shape[1], k=15)

    coef = cp.asarray(theta_sklearn, dtype=cp.float32)
    
    added_varbitset = 0

    for z in range(n):
        new_nodes = []
        for i in Node.varbitset_to_list(Node.universal_varbitset & ~added_varbitset):
            new_node = Node(0, Node.universal_varbitset & ~ (added_varbitset | (1 << i)), coefs = coef.copy())
            new_nodes.append(new_node)
        converged = 0
        converged = parallel_gd2_gpu_kernel(X_gpu, y_gpu, new_nodes, lambda_, verbose=False, tol=tolle, epochs=10000)
        next_var = -1
        worst_obj = math.inf
        for i,j in enumerate(Node.varbitset_to_list(Node.universal_varbitset & ~added_varbitset)):
            if new_nodes[i].lb <= worst_obj:
                next_var = j
                worst_obj = new_nodes[i].lb
                coef = new_nodes[i].coefs
        added_varbitset |= Node.var_to_varbitset(next_var)
        print("Step:",z,"Added:",next_var,"Kept:",Node.varbitset_to_list(added_varbitset))


    quit()

    # Multiple runs to warm up the GPU    
    for numba in range(3):
        # Trying a batch of 60 nodes
        new_nodes = []
        for i in range(60):
            new_node = Node(0, 1 << i, coefs = coef.copy())
            new_node.coefs[i] = 0
            new_nodes.append(new_node)
        # for i in range(10):
        #     new_node = Node(Node.iter_to_varbitset(range(i,i+15)), 0, coefs = coef.copy())
        #     for j in range(15 +i,111):
        #         new_node.coefs[j] = 0
        #     for j in range(i):
        #         new_node.coefs[j] = 0
        #     new_node.is_terminal_leaf()
        #     new_nodes.append(new_node)

        par_time=time.time()
        parallel_gd2_gpu_kernel(X_gpu, y_gpu, new_nodes, lambda_, verbose=False, tol=tolle, epochs=1500)


    
    # sklearn_full_time = time.time()
    # for i in range(10):
    #     # X_sub = X[:, list(range(0,i)) + list(range(i+1,n))]
    #     X_sub = X[:, list(range(i,i+15))]
    #     clf.fit(X_sub, y)
    #     theta_sklearn = clf.coef_.ravel()
    #     z = X_sub @ theta_sklearn
    #     loss = np.logaddexp(0, (1 - 2*y) * z).mean()
    #     reg = 0.5 * lambda_ * np.dot(theta_sklearn, theta_sklearn)
    #     # print(f"obj diff {i}:", new_nodes[i].lb - (loss + reg))
    # # final model
    # X_sub = X[:, list(range(15,n))]
    # clf.fit(X_sub, y)
    # theta_sklearn = clf.coef_.ravel()
    # z = X_sub @ theta_sklearn
    # loss = np.logaddexp(0, (1 - 2*y) * z).mean()
    # reg = 0.5 * lambda_ * np.dot(theta_sklearn, theta_sklearn)
    # print("full sklearn time", time.time() - sklearn_full_time)

    import random

    all_choices = list(range(23)) + list(range(23,n))

    for i in sorted(range(1,n+1), reverse=True):
        # Trying 30 attempts
        for j in range(30):
            # Selecting i nodes to solve
            new_nodes = []
            choices = random.sample(all_choices, i)
            for k in choices:
                new_node = Node(0, 1 << int(k), coefs = coef.copy())
                new_nodes.append(new_node)
            
            par_time=time.time()
            epochs = parallel_gd2_gpu_kernel(X_gpu, y_gpu, new_nodes, lambda_, tol=tolle, epochs=5000)
            par_time = time.time() - par_time
            print(f"MYOGD,{i},{j},{par_time},{epochs},{23 in choices}")