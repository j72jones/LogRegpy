from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective

from LogRegpy.tree.tree import Tree

from LogRegpy.bound_algorithms.sklearn_objective_eval import sklearn_objective_eval
from LogRegpy.bound_algorithms.mosek_objective_eval import MosekObjectiveEval

from LogRegpy.utilities.variable_chooser import VariableChooser

from LogRegpy.variable_choice_heuristics.fractional import Fractional
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.tests.test_logger import TestLogger

dataset_collector = DatasetCollector()
print("Successful data collection:", dataset_collector("IONOSPHERE"))

k = 24

print("Data number of rows:", dataset_collector.rows)
print("Data number of columns:", dataset_collector.n)
print("Goal features:", k)

for solver in ("lbfgs", "MOSEK", "newton-cg", "newton-cholesky", "sag", "saga"):
    
    print("\n\nRunning test for", solver)

    problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)
    test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_ionosphere_{solver}_LeastFractional_None_24.csv")
    test_logger_ins.rewrite_file()

    if solver == "MOSEK":
        test_tree = Tree(
            problem_data.n, 
            problem_data.k, 
            Objective(problem_data, MosekObjectiveEval()),
            Bounder(problem_data, MosekObjectiveEval()),
            Fractional(problem_data, method = "least"),
            initial_upper_bound_strategy=None,
            test_logger=test_logger_ins,
            lower_bounder_fixed_in_agnostic=True
            )
    else:
        test_tree = Tree(
            problem_data.n, 
            problem_data.k, 
            Objective(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False})),
            Bounder(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False})),
            Fractional(problem_data, method = "least"),
            initial_upper_bound_strategy=None,
            test_logger=test_logger_ins
            )

    print("Successful test:", test_tree.solve(timeout = 5))

# To run:
# python -m LogRegpy.tests.test_solvers