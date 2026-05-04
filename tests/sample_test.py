from LogRegpy.utilities.problem_data import ProblemData

from LogRegpy.tree.tree import Tree
from LogRegpy.tree.node import Node


# from LogRegpy.brancher_implementations.mosek_brancher  import MosekBrancher
from LogRegpy.brancher_implementations.gpu_brancher import GPUBrancher
import cupy as cp
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.tests.test_logger import TestLogger

import time
from LogRegpy.tests.hitting_time_tests_uci import greedy_search

dataset_collector = DatasetCollector()
dataset_choice = "MADELON"
print("Collecting data for", dataset_choice)
print("Successful data collection:", dataset_collector(dataset_choice))

for k in range(4,11,2):

    print("Data number of rows:", dataset_collector.rows)
    print("Data number of columns:", dataset_collector.n)
    print("Goal features:", k)
    lamb = 0.001

    # problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)
    # problem_data.X = cp.asarray(problem_data.X, dtype=cp.float32)
    # problem_data.y = cp.asarray(problem_data.y, dtype=cp.float32)

    test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_{dataset_choice}_lamb001_GPU_strong_branching_prescreening_{k}.csv")
    test_logger_ins.rewrite_file()

    start_time = time.time()
    screened_subset, obj = greedy_search(dataset_collector.X, dataset_collector.y,2*k, lam=lamb)

    X_new = dataset_collector.X[:, screened_subset]
    problem_data = ProblemData(X_new, dataset_collector.y, k)
    problem_data.X = cp.asarray(problem_data.X, dtype=cp.float32)
    problem_data.y = cp.asarray(problem_data.y, dtype=cp.float32)

    test_tree = Tree(
        problem_data.n, 
        problem_data.k, 
        GPUBrancher(problem_data, lamb=lamb),
        test_logger=test_logger_ins,
        )

    fixed_in, obj = greedy_search(X_new, dataset_collector.y,k)
    test_tree.best_feasible_node = Node(Node.iter_to_varbitset(fixed_in), 0, lb=obj)
    test_tree.UB = obj

    print("Successful test:", test_tree.solve(timeout = 10000/60, max_iter = 200000, start_time=start_time))
    print("Best subset contains indices:", sorted(Node.varbitset_to_list(test_tree.best_feasible_node.fixed_in)))
    print("Best subset has coefficients:", test_tree.best_feasible_node.coefs[sorted(Node.varbitset_to_list(test_tree.best_feasible_node.fixed_in))])


# To run:
# python -m LogRegpy.tests.sample_test