from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective
from LogRegpy.utilities.variable_chooser import VariableChooser

from LogRegpy.tree.tree import Tree
from LogRegpy.tree.node import Node

from LogRegpy.bound_algorithms.mosek_objective_eval  import MosekObjectiveEval
from LogRegpy.bound_algorithms.sklearn_objective_eval import sklearn_objective_eval
from LogRegpy.bound_algorithms.sklearn_lb import sklearn_lb_eval
from LogRegpy.variable_choice_heuristics.greedy import Greedy
from LogRegpy.variable_choice_heuristics.fractional import Fractional
from LogRegpy.variable_choice_heuristics.maximum_bound_tightening import MaximumBoundTightening
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.initial_upper_bound_heuristics.stepwise_regression import StepwiseRegression
from LogRegpy.tests.test_logger import TestLogger

dataset_collector = DatasetCollector()
dataset_choice = "IONOSPHERE"
print("Collecting data for", dataset_choice)
print("Successful data collection:", dataset_collector(dataset_choice))

k = 6

print("Data number of rows:", dataset_collector.rows)
print("Data number of columns:", dataset_collector.n)
print("Goal features:", k)

problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

test_tree = Tree(
    problem_data.n, 
    problem_data.k, 
    Objective(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False})),
    Bounder(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False})),
    Fractional(problem_data, method="least"),
    initial_upper_bound_strategy=None,
    lower_bounder_fixed_in_agnostic=True
    )

print("Successful test:", test_tree.solve(timeout = 5))
print("Best subset contains indices:", sorted(test_tree.best_feasible_node.fixed_in))

# To run:
# python -m LogRegpy.tests.sample_test