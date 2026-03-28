from LogRegpy.utilities.problem_data import ProblemData

from LogRegpy.tree.tree import Tree
from LogRegpy.tree.node import Node


# from LogRegpy.brancher_implementations.mosek_brancher  import MosekBrancher
from LogRegpy.brancher_implementations.gpu_brancher import GPUBrancher
import cupy as cp
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.tests.test_logger import TestLogger

dataset_collector = DatasetCollector()
dataset_choice = "MYOCARDIAL"
print("Collecting data for", dataset_choice)
print("Successful data collection:", dataset_collector(dataset_choice))

for k in range(1,11,1):

    print("Data number of rows:", dataset_collector.rows)
    print("Data number of columns:", dataset_collector.n)
    print("Goal features:", k)

    problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)
    problem_data.X = cp.asarray(problem_data.X, dtype=cp.float32)
    problem_data.y = cp.asarray(problem_data.y, dtype=cp.float32)

    # test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_IONOSPHERE_GPU_strong_branching_{k}.csv")
    # test_logger_ins.rewrite_file()

    test_tree = Tree(
        problem_data.n, 
        problem_data.k, 
        GPUBrancher(problem_data, lamb=0.1/dataset_collector.rows),
        # test_logger=test_logger_ins
        )

    print("Successful test:", test_tree.solve(timeout = 10000/60, max_iter = 200000))
    print("Best subset contains indices:", sorted(Node.varbitset_to_list(test_tree.best_feasible_node.fixed_in)))


# To run:
# python -m LogRegpy.tests.sample_test