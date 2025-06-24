from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import time
from numpy import setdiff1d


class sklearn_lb_eval:
    def __init__(self, params: dict):
        self.params = params

    def __call__(self, data: ProblemData, node: Node) -> Tuple[float, float]:
        start_time = time.time()
        attributes = setdiff1d(range(data.n), node.fixed_out)
        # Train the logistic regression model
        model = LogisticRegression(**self.params)
        x = data.X[:, attributes]
        model.fit(x, data.y)
        node.coefs = model.coef_.flatten()

        # Calculate score
        return log_loss(data.y, model.predict_proba(x)), time.time() - start_time