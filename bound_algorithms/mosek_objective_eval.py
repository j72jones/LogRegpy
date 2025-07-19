from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from LogRegpy.bound_algorithms.mosek_logistic_model import MosekLogisticModel
from LogRegpy.bound_algorithms.mosek_logistic_regression import logisticRegression

import time
from numpy import setdiff1d


class MosekObjectiveEval:
    def __init__(self, data: ProblemData, lamb=0.0):
        self.model = MosekLogisticModel(data.X, data.y, lamb=lamb)

    def __call__(self, data: ProblemData, node: Node, prev_coefs=None) -> Tuple[float, float]:
        start_time = time.time()
        coefs, obj = self.model.solve(node.fixed_out, prev_coef=prev_coefs)
        node.coefs = {i: coefs[i] for i in setdiff1d(list(range(data.n)), node.fixed_out)}
        return obj, time.time() - start_time
    

if __name__ == "__main__":
    from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
    dataset_collector = DatasetCollector()
    print("Successful data collection:", dataset_collector("MYOCARDIAL"))
    problem_data = ProblemData(dataset_collector.X, dataset_collector.y, 10)
    # Node.n = 500
    # Node.k =10
    node_1 = Node([], [0,1,2,3,4,5,6,7,8])
    # node_2 = Node([48, 204, 226, 296, 323, 424, 430, 431, 475, 481], [])
    # print(node_2.is_terminal_leaf)
    #print(dataset_collector.y)
    start_time = time.time()

    my_eval = MosekObjectiveEval(problem_data)
    # print(my_eval(problem_data, node_1))
    # print(logisticRegression(problem_data.X[:,setdiff1d(range(problem_data.n), [0,1,2,3,4,5,6,7,8])], problem_data.y)[1])
    print(my_eval(problem_data, node_1))
    print()

    # python -m LogRegpy.bound_algorithms.mosek_objective_eval

