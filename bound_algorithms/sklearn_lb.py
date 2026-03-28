from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import time
from numpy import setdiff1d
import numpy as np

class SklearnLogisticModel:
    def __init__(self, params: dict):
        self.params = params
        self.model = LogisticRegression(**self.params)

    def __call__(self, data: ProblemData, node: Node) -> Tuple[float, float]:
        start_time = time.time()
        attributes = setdiff1d(range(data.n), node.fixed_out)
        # Train the logistic regression model
        x = data.X[:, attributes]
        self.model.fit(x, data.y)
        node.coefs = self.model.coef_.flatten()

        # Calculate score
        return SklearnLogisticModel.logistic_l2_objective(node.coefs, x, data.y, 1/self.params["C"]), time.time() - start_time

    def logistic_l2_objective(theta, X, y, lambda_):
        """
        Computes:
        (1/m) * sum log(1 + exp((1 - 2y_i) x_i^T theta))
        + (lambda/2) * ||theta||^2
        """
        m = X.shape[0]

        z = X @ theta
        loss = np.logaddexp(0, (1 - 2*y) * z).mean()
        reg = 0.5 * lambda_ * np.dot(theta, theta)

        return loss + reg