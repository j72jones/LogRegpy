
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


solvers = ("MOSEK", "MOSEK l2", "saga")

datasets = ("SPECT_HEART", "IONOSPHERE", "MADELON", "MYOCARDIAL")

#"WINE","DOROTHEA", 


for solver in solvers:

    print("\n\nRunning tests for", solver)

    for dataset in datasets:
        if dataset != "IONOSPHERE":
            continue

        print("\n\nRunning tests for", dataset)

        dataset_collector = DatasetCollector()
        print("Successful data collection:", dataset_collector(dataset))
        k = 10
        print(f"Selected k for {dataset}:", k)

        problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

        if solver in ("MOSEK", "MOSEK l2"):
            var_selectors = {
                "Least Fractional": Fractional(problem_data, method="least"),
                "Most Fractional": Fractional(problem_data),
                "Random": VariableChooser(problem_data),
                "Most correlated": CorrelationChoice(problem_data),
                "Least correlated": CorrelationChoice(problem_data, method="least"),
                "Greedy Minimization l2": Greedy(problem_data, solver="MOSEK"),
                "Greedy Minimization No Penalty": Greedy(problem_data, penalty=None, solver="MOSEK"),
                "Greedy Maximization l2": Greedy(problem_data, method="max", solver="MOSEK"),
                "Greedy Maximization No Penalty": Greedy(problem_data, penalty=None, method="max", solver="MOSEK")
            }
            heuristics = {
                "Variable Flooring l2": VariableFlooring(problem_data, penalty="l2", solver="MOSEK"),
                "No Heuristic": None,
                
                "Variable Flooring No Penalty": VariableFlooring(problem_data, penalty=None, solver="MOSEK"),
                "Backward Stepwise Regression No Penalty": StepwiseRegression(problem_data, method="backward", solver="MOSEK"),
                "Backward Stepwise Regression l2": StepwiseRegression(problem_data, method="backward", penalty="l2", solver="MOSEK"),
                "Forward Stepwise Regression No Penalty": StepwiseRegression(problem_data, solver="MOSEK"),
                "Forward Stepwise Regression l2": StepwiseRegression(problem_data, penalty="l2", solver="MOSEK"),
            }
        elif solver == "saga":
            var_selectors = {
                "Random": VariableChooser(problem_data),
                "Most correlated": CorrelationChoice(problem_data),
                "Least correlated": CorrelationChoice(problem_data, method="least"),
                "Most Fractional": Fractional(problem_data),
                "Least Fractional": Fractional(problem_data, method="least"),
                "Greedy Minimization l2": Greedy(problem_data),
                "Greedy Minimization l1": Greedy(problem_data, penalty="l1"),
                "Greedy Minimization No Penalty": Greedy(problem_data, penalty=None),
                "Greedy Maximization l2": Greedy(problem_data, method="max"),
                "Greedy Maximization l1": Greedy(problem_data, penalty="l1", method="max"),
                "Greedy Maximization No Penalty": Greedy(problem_data, penalty=None, method="max")
            }
            heuristics = {
                "No Heuristic": None,
                "Variable Flooring l2": VariableFlooring(problem_data, penalty="l2"),
                "Variable Flooring l1": VariableFlooring(problem_data, penalty="l1"),
                "Variable Flooring No Penalty": VariableFlooring(problem_data, penalty=None),
                "Backward Stepwise Regression No Penalty": StepwiseRegression(problem_data, method="backward"),
                "Backward Stepwise Regression l1": StepwiseRegression(problem_data, method="backward", penalty="l1"),
                "Backward Stepwise Regression l2": StepwiseRegression(problem_data, method="backward", penalty="l2"),
                "Forward Stepwise Regression No Penalty": StepwiseRegression(problem_data),
                "Forward Stepwise Regression l1": StepwiseRegression(problem_data, penalty="l1"),
                "Forward Stepwise Regression l2": StepwiseRegression(problem_data, penalty="l2"),
            }

        

        for var_selector in var_selectors.keys():
            print("\n\nRunning tests for", var_selector)


            for heuristic in heuristics.keys():

                print("\n\nRunning test for", heuristic)

                test_logger_ins = TestLogger(f"LogRegpy/tests/test_data/test_{dataset}_{solver}_{var_selector}_{heuristic}_{k}.csv")
                test_logger_ins.rewrite_file()

                if solver == "MOSEK":
                    test_tree = Tree(
                        problem_data.n, 
                        problem_data.k, 
                        Objective(problem_data, MosekObjectiveEval(problem_data)),
                        Bounder(problem_data, MosekObjectiveEval(problem_data)),
                        var_selectors[var_selector],
                        initial_upper_bound_strategy=heuristics[heuristic],
                        test_logger=test_logger_ins,
                        lower_bounder_fixed_in_agnostic=True
                        )
                elif solver == "MOSEK l2":
                    test_tree = Tree(
                        problem_data.n, 
                        problem_data.k, 
                        Objective(problem_data, MosekObjectiveEval(problem_data, lamb=1)),
                        Bounder(problem_data, MosekObjectiveEval(problem_data, lamb=1)),
                        var_selectors[var_selector],
                        initial_upper_bound_strategy=heuristics[heuristic],
                        test_logger=test_logger_ins,
                        lower_bounder_fixed_in_agnostic=True
                        )
                else:
                    test_tree = Tree(
                        problem_data.n, 
                        problem_data.k, 
                        Objective(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False, "solver": solver})),
                        Bounder(problem_data, sklearn_objective_eval(params={"penalty": None, "fit_intercept": False,  "solver": solver})),
                        var_selectors[var_selector],
                        initial_upper_bound_strategy=heuristics[heuristic],
                        test_logger=test_logger_ins
                        )

                print("Successful test:", test_tree.solve(timeout = 10000))


# To run:
# python -m LogRegpy.tests.all_regular_tests


