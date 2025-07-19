
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

queue = [['SPECT_HEART', 2], ['SPECT_HEART', 4], ['SPECT_HEART', 6], ['SPECT_HEART', 8], ['SPECT_HEART', 13], ['SPECT_HEART', 15], ['SPECT_HEART', 17], ['SPECT_HEART', 19], ['MYOCARDIAL', 11], ['MYOCARDIAL', 22], ['MYOCARDIAL', 33], ['MYOCARDIAL', 44], ['MYOCARDIAL', 55], ['MYOCARDIAL', 66], ['MYOCARDIAL', 77], ['MYOCARDIAL', 88], ['MYOCARDIAL', 99], ['MADELON', 50], ['MADELON', 100], ['MADELON', 150], ['MADELON', 200], ['MADELON', 300], ['MADELON', 350], ['MADELON', 400], ['MADELON', 450]]

#"WINE","DOROTHEA", 

for dataset,k in queue:

    print("\n\nRunning tests for", dataset)

    dataset_collector = DatasetCollector()
    print("Successful data collection:", dataset_collector(dataset))
    print(f"Selected k for {dataset}:", k)

    problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

    # heuristics = {
    #     "Backward Stepwise Regression No Penalty": StepwiseRegression(problem_data, method="backward", solver="MOSEK"),
    #     "Forward Stepwise Regression No Penalty": StepwiseRegression(problem_data, solver="MOSEK"),
    # }

    test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_{dataset}_MOSEK_Least Fractional_Variable Flooring l2_{k}.csv")
    test_logger_ins.rewrite_file()

    test_tree = Tree(
        problem_data.n, 
        problem_data.k, 
        Objective(problem_data, MosekObjectiveEval(problem_data)),
        Bounder(problem_data, MosekObjectiveEval(problem_data)),
        Fractional(problem_data, method="least"),
        initial_upper_bound_strategy=VariableFlooring(problem_data, penalty="l2", solver="MOSEK"),
        test_logger=test_logger_ins,
        lower_bounder_fixed_in_agnostic=True
        )

    print("Successful test:", test_tree.solve(timeout = 10000))


# To run:
# python -m LogRegpy.tests.final_sampling_tests


