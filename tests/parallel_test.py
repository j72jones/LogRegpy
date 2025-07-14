from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective
from LogRegpy.utilities.variable_chooser import VariableChooser

from LogRegpy.tree.parallel_tree import ParallelTree

from LogRegpy.bound_algorithms.mosek_objective_eval  import MosekObjectiveEval
from LogRegpy.bound_algorithms.sklearn_objective_eval import sklearn_objective_eval
from LogRegpy.bound_algorithms.sklearn_lb import sklearn_lb_eval
from LogRegpy.variable_choice_heuristics.greedy import Greedy
from LogRegpy.variable_choice_heuristics.fractional import Fractional
from LogRegpy.variable_choice_heuristics.maximum_bound_tightening import MaximumBoundTightening
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.initial_upper_bound_heuristics.stepwise_regression import StepwiseRegression
from LogRegpy.tests.test_logger import TestLogger

if __name__ == "__main__":

    dataset_collector = DatasetCollector()
    print("Successful data collection:", dataset_collector("IONOSPHERE"))

    k = 24

    print("Data number of rows:", dataset_collector.rows)
    print("Data number of columns:", dataset_collector.n)
    print("Goal features:", k)



    problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)
    # test_logger_ins = TestLogger("LogRegpy/tests/test_ionosphere_mosek_mosek_fractional_None_24.csv")
    # test_logger_ins.rewrite_file()

    test_tree = ParallelTree(
        problem_data.n, 
        problem_data.k, 
        Objective(problem_data, MosekObjectiveEval()),
        Bounder(problem_data, MosekObjectiveEval()),
        Fractional(problem_data, method="least"),
        4,
        initial_upper_bound_strategy=None,
        lower_bounder_fixed_in_agnostic=True
        )

    print("\n\n\n Successful test:", test_tree.solve(timeout = 5))

#print(len(min(test_tree.feasible_leaves).fixed_out))

# To run:
# python -m LogRegpy.tests.parallel_test