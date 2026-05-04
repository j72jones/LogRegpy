import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.tree import Tree
from LogRegpy.tree.node import Node
from LogRegpy.brancher_implementations.gpu_brancher import GPUBrancher
import cupy as cp
from ucimlrepo import fetch_ucirepo 
import math
import time

rng = np.random.default_rng(2)
lamb=1e-5

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_objective(beta, X, y, lam=lamb):
    margin = (1 - 2 * y) * (X @ beta)  # (n,)
    logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
    logistic_loss = np.mean(logistic_terms)

    # L2 regularization
    reg = (lam / 2) * np.dot(beta, beta)

    return logistic_loss + reg

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

def greedy_search(X, y, k, lam=lamb):
    n, p = X.shape
    A = []
    Aset = set()
    beta = fit_subset_logistic_ridge(X, y, A, lam=lam)
    for t in range(1, k+1):
        p_hat = sigmoid(X @ beta)
        scores = np.abs(np.mean(((p_hat - y)[:, None]) * X, axis=0))
        if A:
            scores[A] = -np.inf
        j_star = int(np.argmax(scores))
        A.append(j_star)
        Aset.add(j_star)
        beta = fit_subset_logistic_ridge(X, y, A, lam=lam)
    return A, logistic_objective(beta, X, y)

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
        k = int(rng.integers(3, min(9, p)))

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



def run_experiments(uci_id):
    results = []
    results.append(["n", "p", "k", "uci_id", "tau"])
    k=3
    unique_vals = []
    dataset = fetch_ucirepo(id=uci_id)
    X = dataset.data.features.fillna(0) # type: ignore
    X = pd.get_dummies(X).to_numpy().astype(float)
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)  # avoid divide-by-zero
    X = (X - X_mean) / X_std
    y = dataset.data.targets.to_numpy().ravel() # type: ignore
    unique_vals = np.unique(y)
    if len(unique_vals) != 2:
        print(f"Too many unique values in this dataset targets")
        print(unique_vals)
    y = (y == unique_vals[1]).astype(int)
    n, p = X.shape

    for k in range(2,13):
        
        print(f"\nStarting: {uci_id}: {n},{p},{k}")

        # --- run B&B ---
        problem_data = ProblemData(X, y, k)
        problem_data.X = cp.asarray(problem_data.X, dtype=cp.float32)
        problem_data.y = cp.asarray(problem_data.y, dtype=cp.float32)
        C = 1.0 / (lamb * n)
        test_tree = Tree(
            problem_data.n, 
            problem_data.k, 
            GPUBrancher(problem_data,
                            lamb=lamb)
            )

        fixed_in, obj = greedy_search(X,y,k)
        test_tree.best_feasible_node = Node(Node.iter_to_varbitset(fixed_in), 0, lb=obj)
        test_tree.UB = obj

        start_solve_time = time.time()
        if test_tree.solve(timeout = 30, max_iter = 200000, verbose=False):
            print(f"Iteration {test_tree.num_iter} | gap = {test_tree.gap:.4f} | Open Subproblems: {len(test_tree.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {test_tree.remaining_tree_size:,} | Running Time: {time.time() - start_solve_time:.2f} seconds")
            print(f"Successful test: {uci_id}: {n},{p},{k}")
        else:
            print(f"Iteration {test_tree.num_iter} | gap = {test_tree.gap:.4f} | Open Subproblems: {len(test_tree.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {test_tree.remaining_tree_size:,} | Running Time: {time.time() - start_solve_time:.2f} seconds")
            print(f"Unsuccessful test: {uci_id}: {n},{p},{k}")
        support = set(Node.varbitset_to_list(test_tree.best_feasible_node.fixed_in))

        # --- run greedy ---
        start_greedy_time = time.time()
        tau,_ = greedy_path_hitting_time(X, y, support)
        print("Greedy Time:", time.time() - start_greedy_time, "Greedy tau:", tau)

        results.append([
            n,
            p,
            k,
            uci_id,
            tau
        ])

        pd.DataFrame(results).to_csv(f"results_UCI_gpu_1_{int(args.uci_id)}.csv")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--uci_id", type=float)

    args = parser.parse_args()

    # uci_ids = [
    #     # 17,   # Breast Cancer Wisconsin (Diagnostic)
    #     # 45,   # Heart Disease
    #     # 53,   # Ionosphere
    #     # 60,   # Sonar
    #     # 73,   # Mushrooms
    #     94,   # Spambase
    #     109,  # Statlog (German Credit)
    #     144,  # Statlog (Australian Credit)
    #     145,  # Banknote Authentication
    #     151,  # Connectionist Bench (Sonar, Mines vs Rocks alt)
    #     222,  # Bank Marketing
    #     267,  # Parkinsons
    #     275,  # Breast Cancer Coimbra
    #     350,  # Default of Credit Card Clients
    #     # 360,  # Online Shoppers Intention
    #     380,  # Phishing Websites
    #     451,  # Heart Failure Clinical Records
    #     # 463,  # QSAR Biodegradation
    #     468,  # Electrical Grid Stability (binary version)
    #     # 471,  # Climate Model Simulation Crashes
    #     477,  # Real Estate Valuation (binary variants used)
    #     492,  # Credit Approval
    #     # 507,  # HCV Data
    #     519,  # Travel Review Ratings (binary subset)
    #     545,  # Hepatitis
    #     563,  # Cervical Cancer (Risk Factors)
    #     571,  # Early Stage Diabetes Risk Prediction
    #     579,  # Heart Disease (newer processed)
    #     601,  # Raisin Dataset (binary classification)
    #     602,  # Rice (Cammeo vs Osmancik)
    # ]

    results = run_experiments(
        int(args.uci_id)
    )

    # save results
    pd.DataFrame(results).to_csv(f"results_UCI_gpu_1e-5_{int(args.uci_id)}.csv")