
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
from LogRegpy.variable_choice_heuristics.correlation_choice import CorrelationChoice
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
from LogRegpy.initial_upper_bound_heuristics.stepwise_regression import StepwiseRegression
from LogRegpy.initial_upper_bound_heuristics.variable_flooring import VariableFlooring
from LogRegpy.tests.test_logger import TestLogger

import pandas as pd


solvers = ("MOSEK", "MOSEK l2", "saga")

datasets = ("SPECT_HEART", "IONOSPHERE", "MADELON", "MYOCARDIAL")

#"WINE","DOROTHEA", 



dataset_collector = DatasetCollector()
print("Successful data collection:", dataset_collector("IONOSPHERE"))
for k in sorted(range(1,dataset_collector.n), reverse=False):

    print(f"Testing k:", k)

    problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

    var_selectors = {
        "Least Fractional": Fractional(problem_data, method="least"),
        #"Most Fractional": Fractional(problem_data),
    }
    heuristics = {
        "Variable Flooring l2": VariableFlooring(problem_data, penalty="l2", solver="MOSEK"),
        #"No Heuristic": None,
        "Variable Flooring No Penalty": VariableFlooring(problem_data, penalty=None, solver="MOSEK"),
        "Backward Stepwise Regression No Penalty": StepwiseRegression(problem_data, method="backward", solver="MOSEK"),
        "Backward Stepwise Regression l2": StepwiseRegression(problem_data, method="backward", penalty="l2", solver="MOSEK"),
        "Forward Stepwise Regression No Penalty": StepwiseRegression(problem_data, solver="MOSEK"),
        "Forward Stepwise Regression l2": StepwiseRegression(problem_data, penalty="l2", solver="MOSEK"),
    }

    heuristic_data = []
    for heuristic in heuristics.keys():
        print("\n\nRunning test for", heuristic)
        heuristic_data.append([heuristic] + list(heuristics[heuristic]()))

    pd.DataFrame(heuristic_data, columns=["Heuristic", "UB", "Time", "Selected Variables"]).to_csv(f"/Users/jamesjones/Research_Weijun/LogRegpy/tests/heuristic_test_data/IONOSPHERE_{k}", index=False)

    # for var_selector in var_selectors.keys():
    #     print("\n\nRunning tests for", var_selector)

    #     test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_IONOSPHERE_MOSEK_{var_selector}_No Heuristic_{k}.csv")
    #     test_logger_ins.rewrite_file()

    #     test_tree = Tree(
    #         problem_data.n, 
    #         problem_data.k, 
    #         Objective(problem_data, MosekObjectiveEval(problem_data)),
    #         Bounder(problem_data, MosekObjectiveEval(problem_data)),
    #         var_selectors[var_selector],
    #         initial_upper_bound_strategy=None,
    #         test_logger=test_logger_ins,
    #         #lower_bounder_fixed_in_agnostic=True
    #         )


    #     print("Successful test:", test_tree.solve(timeout = 30))
    



# To run:
# python -m LogRegpy.tests.all_ionosphere_tests


