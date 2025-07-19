from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective
from LogRegpy.utilities.variable_chooser import VariableChooser

from LogRegpy.tree.tree_greedy import TreeGreedy
from LogRegpy.tree.node import Node

from LogRegpy.bound_algorithms.mosek_objective_eval  import MosekObjectiveEval
from LogRegpy.bound_algorithms.sklearn_objective_eval import sklearn_objective_eval
from LogRegpy.bound_algorithms.sklearn_lb import sklearn_lb_eval
from LogRegpy.variable_choice_heuristics.greedy import Greedy
from LogRegpy.variable_choice_heuristics.fractional import Fractional
from LogRegpy.variable_choice_heuristics.maximum_bound_tightening import MaximumBoundTightening
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.initial_upper_bound_heuristics.direct_submission import DirectSubmission
from LogRegpy.tests.test_logger import TestLogger

dataset_collector = DatasetCollector()
print("Successful data collection:", dataset_collector("MADELON"))

k = 10

print("Data number of rows:", dataset_collector.rows)
print("Data number of columns:", dataset_collector.n)
print("Goal features:", k)



problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)
test_logger_ins = TestLogger("LogRegpy/tests/test_data/test_greedy_MADELON_mosek_mosek_No Heuristic_10.csv")
test_logger_ins.rewrite_file()

test_tree = TreeGreedy(
    problem_data.n, 
    problem_data.k, 
    Objective(problem_data, MosekObjectiveEval(problem_data)),
    Bounder(problem_data, MosekObjectiveEval(problem_data)),
    initial_upper_bound_strategy=DirectSubmission(problem_data, fixed_in = [48, 204, 226, 296, 323, 424, 430, 431, 475, 481]),
    lower_bounder_fixed_in_agnostic=True,
    test_logger=test_logger_ins
    )

print("Successful test:", test_tree.solve(timeout = 100000))

#print(len(min(test_tree.feasible_leaves).fixed_out))

# To run:
# python -m LogRegpy.tests.greedy_test