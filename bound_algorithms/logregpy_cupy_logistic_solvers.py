import cupy as cp
from LogRegpy.tree.node import Node
import time
from typing import Optional


def sigmoid(z):
        return 1.0 / (1.0 + cp.exp(-z))

def parallel_gd(X: cp.ndarray,
                 y: cp.ndarray,
                 nodes: list[Node],
                 lr0: Optional[float] = 0.001,
                 epochs: Optional[int] = 5,
                 tol: Optional[float] = 1e-8,
                 verbose: Optional[bool] = False):
    """
    Parallel SGD for logistic regression with leave-one-feature-out models,
    with early stopping based on log loss convergence.

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

    # Build per-model masks and warm-start coefficients
    masks = [cp.array(Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)) for node in nodes]
    W_all = [node.coefs for node in nodes]

    objs = [cp.inf for _ in range(len(nodes))]
    new_objs = [cp.inf for _ in range(len(nodes))]
    converged = [False for _ in range(len(nodes))]

    for epoch in range(1, 1+epochs):
        # learning rate decay
        lr = lr0 / epoch
        for m, mask in enumerate(masks):
            if converged[m]:
                continue
            X_sub = X[:, mask]
            logits = X_sub @ W_all[m][mask]
            preds = sigmoid(logits)
            grad = (X_sub.T @ (preds - y))
            # write back to the full-sized vector at masked indices
            W_all[m][mask] -= lr * grad

            # recompute loss at updated weights
            logits = X_sub @ W_all[m][mask]
            margin = (1 - 2*y) * logits
            logistic_terms = cp.where(
                margin > 0,
                margin + cp.log1p(cp.exp(-margin)),
                cp.log1p(cp.exp(margin))
            )
            new_loss = cp.sum(logistic_terms)
            new_objs[m] = new_loss
            if converged[m] and cp.abs(new_objs[m] - objs[m])/new_objs[m] > tol:
                converged[m] = False
            objs[m] = new_objs[m]
        if verbose:
            print(f"Epoch {epoch+1}, converged={converged}")
        if all(converged) and epoch > 5:
            if verbose:
                print(f"Converged at epoch {epoch+1}")
            break

    for j in range(len(nodes)):
        nodes[j].lb = new_objs[j]

    return nodes

def single_gd(X: cp.ndarray,
              y: cp.ndarray,
              warm_start_coefs: Optional[cp.ndarray] = None,
              lr0: Optional[float] = 0.9,
              epochs: Optional[int] = 1500,
              tol: Optional[float] = 1e-8,
              verbose: Optional[bool] = False):
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
    if warm_start_coefs:
        coefs = warm_start_coefs
    else:
        coefs = cp.zeros_like(X[0]) # array with shape equal to no. of features
    
    # Performing Gradient Descent Optimization for every epoch
    loss = cp.inf
    for epoch in range(1,epochs+1):
        LR = lr0 / (epoch)
        z = X @ coefs     # shape (n,)
        pred = sigmoid(z)
        grad = X.T @ (pred-y)
        coefs -= LR * grad
        # Recompute loss at updated weights
        z = X @ coefs
        margin = (1 - 2*y) * z
        logistic_terms = cp.where(
            margin > 0,
            margin + cp.log1p(cp.exp(-margin)),
            cp.log1p(cp.exp(margin))
        )
        new_loss = cp.sum(logistic_terms)
        if cp.abs(new_loss - loss)/new_loss < tol:
            break
        loss = new_loss
    return coefs, new_loss


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
    if warm_start_coefs:
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