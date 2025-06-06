from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective

from LogRegpy.tree.tree import Tree

from LogRegpy.bound_research_algorithms.sklearn_objective_eval import sklearn_objective_eval
from LogRegpy.bound_research_algorithms.sklearn_lb import sklearn_lb_eval
from LogRegpy.variable_choice_research_algorithms.greedy import Greedy
from LogRegpy.tests.datasets.dataset_collector import DatasetCollector


dataset_collector = DatasetCollector()
print("Successful data collection:", dataset_collector("SPECT_HEART"))

k = dataset_collector.n * 2 // 3

print("Data number of rows:", dataset_collector.rows)
print("Data number of columns:", dataset_collector.n)
print("Goal features:", k)



problem_data = ProblemData(dataset_collector.X, dataset_collector.y, k)

test_tree = Tree(
    problem_data.n, 
    problem_data.k, 
    Objective(problem_data, sklearn_objective_eval),
    Bounder(problem_data, sklearn_lb_eval),
    Greedy(problem_data)
    )

print("Successful test:", test_tree.solve())

# To run:
# python -m LogRegpy.tests.test_6_jun_2025