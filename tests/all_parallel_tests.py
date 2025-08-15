
from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective
from LogRegpy.utilities.variable_chooser import VariableChooser

from LogRegpy.tree.parallel_tree import ParallelTree
from LogRegpy.tree.node import Node

from LogRegpy.bound_algorithms.mosek_objective_eval  import MosekObjectiveEval
from LogRegpy.bound_algorithms.mosek_parallel_objective_eval  import MosekParallelObjectiveEval
from LogRegpy.bound_algorithms.sklearn_objective_eval import sklearn_objective_eval
from LogRegpy.bound_algorithms.sklearn_lb import sklearn_lb_eval
from LogRegpy.variable_choice_heuristics.greedy import Greedy
from LogRegpy.variable_choice_heuristics.fractional import Fractional
from LogRegpy.variable_choice_heuristics.correlation_choice import CorrelationChoice
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.initial_upper_bound_heuristics.stepwise_regression import StepwiseRegression
from LogRegpy.initial_upper_bound_heuristics.variable_flooring import VariableFlooring
from LogRegpy.tests.test_logger import TestLogger
import os

if __name__ == "__main__":

    num_processes = os.cpu_count()
    print("trying up to", num_processes)

    dataset_collector = DatasetCollector()
    print("Successful data collection:", dataset_collector("IONOSPHERE"))
    for k in [2,6,10,14]:

        print(f"Testing k:", k)

        problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

        

        for j in [7,5,3,1]:
            test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_parallel{j}_IONOSPHERE_MOSEK_Least Fractional_No Heuristic_{k}.csv")
            test_logger_ins.rewrite_file()
            test_tree = ParallelTree(
                problem_data.n, 
                problem_data.k, 
                Objective(problem_data, MosekParallelObjectiveEval(problem_data)),
                Bounder(problem_data, MosekObjectiveEval(problem_data)),
                Fractional(problem_data, method="least"),
                test_logger=test_logger_ins,
                initial_upper_bound_strategy=None,
                lower_bounder_fixed_in_agnostic=True
            )

            print("Successful test:", test_tree.solve(j, timeout = 5), "\n")


# To run:
# python -m LogRegpy.tests.all_parallel_tests


