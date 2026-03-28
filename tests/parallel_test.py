from LogRegpy.utilities.problem_data import ProblemData

from LogRegpy.tree.parallel_tree import ParallelTree
from LogRegpy.tree.node import Node

from LogRegpy.initial_upper_bound_heuristics.mosek_upper_bounder import MOSEKUpperBounder
from LogRegpy.brancher_implementations.mosek_brancher import MosekBrancher
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.tests.test_logger import TestLogger

for dataset_choice in ("MYOCARDIAL", "MADELON"):
    dataset_collector = DatasetCollector()
    print("Collecting data for", dataset_choice)
    print("Successful data collection:", dataset_collector(dataset_choice))
    print("Data number of rows:", dataset_collector.rows)
    print("Data number of columns:", dataset_collector.n)
    
    for k in range(4,11,2):
        print("Goal features:", k)
        problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

        for j in (2, 4, 8, 16):
            print("Testing", j, "branchers")

            method="smallest_coefficient"
            ub_method = "forward_selection"

            test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/ds-{dataset_choice}__k-{k}__method-{method}__ub-{ub_method}__nb-{j}.csv")
            test_logger_ins.rewrite_file()

            test_tree = ParallelTree(
                problem_data.n, 
                problem_data.k, 
                MosekBrancher(problem_data, method="smallest_coefficient"),
                initial_upper_bound_strategy=MOSEKUpperBounder(problem_data, method="forward_selection")
                )
            
            print("Successful test:", test_tree.solve(j, timeout = 1003/60))
            print("Best subset contains indices:", sorted(Node.varbitset_to_list(test_tree.best_feasible_node.fixed_in)))


# To run:
# python -m LogRegpy.tests.sample_test