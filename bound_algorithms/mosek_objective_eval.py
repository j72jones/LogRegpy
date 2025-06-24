from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from LogRegpy.bound_algorithms.mosek_logistic_regression import logisticRegression
import time
from numpy import setdiff1d


class MosekObjectiveEval:
    def __init__(self):
        pass

    def __call__(self, data: ProblemData, node: Node) -> Tuple[float, float]:
        start_time = time.time()
        attributes = setdiff1d(range(data.n), node.fixed_out)
        x = data.X[:, attributes]
        coefs, obj = logisticRegression(x, data.y)
        node.coefs = {attr: coefs[i] for i,attr in enumerate(attributes)}
        return obj, time.time() - start_time
    

if __name__ == "__main__":
    from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
    dataset_collector = DatasetCollector()
    print("Successful data collection:", dataset_collector("IONOSPHERE"))
    #print(dataset_collector.y)
    coefs, obj = logisticRegression(dataset_collector.X, dataset_collector.y)
    print(obj)

    # python -m LogRegpy.bound_algorithms.mosek_objective_eval