from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective

from LogRegpy.tree.tree import Tree
from LogRegpy.tree.node import Node

from LogRegpy.bound_algorithms.sklearn_objective_eval import sklearn_objective_eval
from LogRegpy.bound_algorithms.sklearn_lb import sklearn_lb_eval
from LogRegpy.variable_choice_heuristics.greedy import Greedy
from LogRegpy.variable_choice_heuristics.fractional import Fractional
from LogRegpy.variable_choice_heuristics.maximum_bound_tightening import MaximumBoundTightening
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.initial_upper_bound_heuristics.stepwise_regression import StepwiseRegression
from LogRegpy.tests.test_logger import TestLogger

dataset_collector = DatasetCollector()
print("Successful data collection:", dataset_collector("IONOSPHERE"))

k = 24

print("Data number of rows:", dataset_collector.rows)
print("Data number of columns:", dataset_collector.n)
print("Goal features:", k)


problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

heuristics = {
    "Backward Stepwise Regression": StepwiseRegression(problem_data, "backward"),
    "Forward Stepwise Regression": StepwiseRegression(problem_data),
    "No Heuristic": None
}

for heuristic in heuristics.keys():
    print("Trying:", heuristic)
    test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_ionosphere_sklearn_sklearn_LeastFractional_{heuristic}_24.csv")
    test_logger_ins.rewrite_file()

    test_tree = Tree(
        problem_data.n, 
        problem_data.k, 
        Objective(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False})),
        Bounder(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False})),
        Fractional(problem_data, method="least"),
        initial_upper_bound_strategy=heuristics[heuristic],
        test_logger=test_logger_ins
        )

    print("Successful test:", test_tree.solve(timeout = 5))

    print(len(min(test_tree.feasible_leaves).fixed_out))

# To run:
# python -m LogRegpy.tests.test_initial_UB_heuristics