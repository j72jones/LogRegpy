import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.tree import Tree
from LogRegpy.tree.node import Node
from LogRegpy.brancher_implementations.gpu_brancher import GPUBrancher
from LogRegpy.bound_algorithms.logregpy_cupy_logistic_solvers import parallel_gd2_gpu_kernel, generate_logistic_data
import cupy as cp
import math
import time

rng = np.random.default_rng(2)
lamb=0.001

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_objective(beta, X, y, lam=lamb):
    # Linear term: X @ beta gives (n,)
    z = X @ beta  # shape (n,)

    # Transform y from {0,1} to {+1, -1} via (1 - 2y)
    margin = (1 - 2 * y) * z

    # Numerically stable log(1 + exp(x)) using np.logaddexp
    loss = np.mean(np.logaddexp(0, margin))

    # L2 regularization
    reg = (lam / 2) * np.dot(beta, beta)

    return loss + reg

def fit_subset_logistic_ridge(X, y, subset, lam=lamb):
    n, p = X.shape
    beta = np.zeros(p)
    subset = list(subset)
    if len(subset) == 0:
        return beta
    C = 1.0 / (lam * n)
    clf = LogisticRegression(
        C=C,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-9,
    )
    clf.fit(X[:, subset], y)
    beta[subset] = clf.coef_.ravel()
    return beta

def greedy_path_hitting_time(X, y, support, lam=lamb):
    n, p = X.shape
    support = set(support)
    true_obj = logistic_objective(fit_subset_logistic_ridge(X,y,list(support)), X, y)
    A = []
    Aset = set()
    beta = fit_subset_logistic_ridge(X, y, A, lam=lam)
    for t in range(1, p + 1):
        p_hat = sigmoid(X @ beta)
        scores = np.abs(np.mean(((p_hat - y)[:, None]) * X, axis=0))
        if A:
            scores[A] = -np.inf
        j_star = int(np.argmax(scores))
        A.append(j_star)
        Aset.add(j_star)
        beta = fit_subset_logistic_ridge(X, y, A, lam=lam)
        if support.issubset(Aset) or (len(Aset) >= len(support) and true_obj >= logistic_objective(beta, X, y)):
            return t, A
    return p, A

def independent_scenario(n=1200, p=100, s_star=4):
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    support = set(range(s_star))
    beta[list(range(s_star))] = np.array([1.3, 1.1, 0.9, 0.8])
    y = rng.binomial(1, sigmoid(X @ beta))
    return X, y, support

def suppressor_scenario(n=1000, p=80):
    Z = rng.normal(size=(n, 3))
    X = rng.normal(size=(n, p))
    x0 = Z[:, 0]
    x1 = Z[:, 1]
    rho = 0.9
    x2 = rho * x0 + np.sqrt(1.0 - rho**2) * Z[:, 2]
    X[:, 0] = x0
    X[:, 1] = x1
    X[:, 2] = x2
    beta = np.zeros(p)
    beta[[0, 1, 2]] = np.array([2.0, 1.0, -2.0])
    support = {0, 1, 2}
    y = rng.binomial(1, sigmoid(X @ beta))
    return X, y, support

def sample_problem(rng):
    while True:
        p = int(rng.integers(30, 80))
        k = int(rng.integers(3, min(13, p))) # was set to 9, skewing results

        # bias s toward k
        s = max(1, int(rng.normal(loc=0.7 * k, scale=1)))
        s = min(s, k)

        # sample n based on theory
        n_min = int(5 * s * math.log(p))
        n_max = int(20 * s * math.log(p))
        n = int(rng.integers(max(50, n_min), max(n_min + 1, n_max)))

        # feasibility check
        if math.comb(p, k) > 5e7:
            continue

        return n, p, s, k



def run_experiments(time_limit_hours, rho, beta_scale, corr_type, seed=0):
    rng = np.random.default_rng(seed)

    time_limit = time_limit_hours * 3600
    start_time = time.time()

    results = []
    results.append(["n", "p", "s", "k", "rho", "beta_scale", "corr_type", "tau"])
    while time.time() - start_time < time_limit:
        n, p, s, k = sample_problem(rng)
        print("Starting:", n,p,s,k)

        X, y, beta_true, support = generate_logistic_data(
            N=n,
            n=p,
            m=s,
            rho=rho,
            beta_scale=beta_scale,
            corr_type=corr_type,
            rng=rng
        )

        # --- run B&B ---
        problem_data = ProblemData(X, y, k)
        problem_data.X = cp.asarray(problem_data.X, dtype=cp.float32)
        problem_data.y = cp.asarray(problem_data.y, dtype=cp.float32)

        test_tree = Tree(
            problem_data.n, 
            problem_data.k, 
            GPUBrancher(problem_data, lamb=lamb),
            )

        start_solve_time = time.time()
        if test_tree.solve(timeout = 17, max_iter = 200000, verbose=False):
            print(f"Iteration {test_tree.num_iter} | Running Time: {time.time() - start_solve_time:.2f} seconds")
            print("Successful test:", n,p,s,k)
        else:
            print(f"Iteration {test_tree.num_iter} | gap = {test_tree.gap:.4f} | Open Subproblems: {len(test_tree.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {test_tree.remaining_tree_size:,} | Running Time: {time.time() - start_solve_time:.2f} seconds")
            print("Unsuccessful test:", n,p,s,k)
        support = set(Node.varbitset_to_list(test_tree.best_feasible_node.fixed_in))

        # --- run greedy ---
        start_greedy_time = time.time()
        tau,_ = greedy_path_hitting_time(X, y, support)
        print("Greedy Time:", time.time() - start_greedy_time, "Greedy tau:", tau)

        results.append([
            n,
            p,
            s,
            k,
            rho,
            beta_scale,
            corr_type,
            tau
        ])

        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(f"results_001_{args.corr_type}_rho{args.rho}_beta{args.beta_scale}.csv")
    return results

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rho", type=float, required=True)
parser.add_argument("--beta_scale", type=float, required=True)
parser.add_argument("--corr_type", type=str, required=True)
parser.add_argument("--hours", type=float, default=8)

args = parser.parse_args()

results = run_experiments(
    time_limit_hours=args.hours,
    rho=args.rho,
    beta_scale=args.beta_scale,
    corr_type=args.corr_type
)

# save results
pd.DataFrame(results).to_csv(f"results_001_{args.corr_type}_rho{args.rho}_beta{args.beta_scale}.csv")