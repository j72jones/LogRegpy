from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import time
from numpy import setdiff1d
import numpy as np


class sklearn_objective_eval:
    def __init__(self, params: dict):
        self.params = params

    def __call__(self, data: ProblemData, node: Node, prev_coefs=None) -> Tuple[float, float]:
        start_time = time.time()
        attributes = setdiff1d(range(data.n), node.fixed_out)
        # Train the logistic regression model
        model = LogisticRegression(**self.params)
        x = data.X[:, attributes]
        model.fit(x, data.y)
        node.coefs = model.coef_.flatten()

        margin = (1 - 2 * data.y) * (x @ node.coefs)  # (n,)
        logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
        logistic_loss = np.sum(logistic_terms)

        # Calculate score
        return logistic_loss, time.time() - start_time
    
if __name__ == "__main__":
    
    from LogRegpy.tests.datasets.dataset_collector import DatasetCollector
    dataset_collector = DatasetCollector()
    print("Successful data collection:", dataset_collector("IONOSPHERE"))
    model = LogisticRegression(penalty=None, fit_intercept=False)
    model.fit(dataset_collector.X, dataset_collector.y)
    print(log_loss(dataset_collector.y, model.predict_proba(dataset_collector.X)))
    y = dataset_collector.y
    X = dataset_collector.X
    theta = model.coef_.flatten()
    margin = (1 - 2 * y) * (X @ theta)  # (n,)
    logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
    logistic_loss = np.sum(logistic_terms)
    print(logistic_loss)