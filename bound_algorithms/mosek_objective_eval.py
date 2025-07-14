from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from LogRegpy.bound_algorithms.mosek_logistic_model import MosekLogisticModel

import time
from numpy import setdiff1d


class MosekObjectiveEval:
    def __init__(self, data: ProblemData, lamb=0.0):
        self.model = MosekLogisticModel(data.X, data.y, lamb=lamb)

    def __call__(self, data: ProblemData, node: Node, prev_coefs=None) -> Tuple[float, float]:
        start_time = time.time()
        coefs, obj = self.model.solve(node.fixed_out, prev_coef=prev_coefs)
        node.coefs = {attr: coefs[i] for i,attr in enumerate(setdiff1d(list(range(data.n)), node.fixed_out))}
        return obj, time.time() - start_time
    

if __name__ == "__main__":
    from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
    dataset_collector = DatasetCollector()
    print("Successful data collection:", dataset_collector("MADELON"))
    problem_data = ProblemData(dataset_collector.X, dataset_collector.y, 10)
    node_1 = Node([1,2], [])
    node_2 = Node([], [])
    #print(dataset_collector.y)
    start_time = time.time()

    my_eval = MosekObjectiveEval(problem_data)
    my_eval(problem_data, node_1)
    my_eval(problem_data, node_2)


    # python -m LogRegpy.bound_algorithms.mosek_objective_eval