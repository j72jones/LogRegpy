from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import math
import time
import numpy as np

class Greedy(VariableChooser):
    def __init__(self, data: ProblemData) -> None:
        self.data = data

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        attributes = setdiff1d(range(self.data.n), node.fixed_out)
        beta = node.model.coef_.flatten()
        x = self.data.X[:, attributes]
        p = node.model.predict_proba(x)
        
        # Compute gradient w.r.t. each unselected variable
        gradients = {}
        residual = self.data.y - p
        for j in setdiff1d(range(self.data.n), node.fixed_out + node.fixed_in):
            xj = x[:, j]
            grad_j = np.abs(np.dot(xj, residual))
            gradients[j] = grad_j
        
        # Select variable with largest absolute gradient
        best_feature = max(gradients.items(), key=lambda x: x[1])[0]
        return best_feature
        return (best_var, start_time - time.time())

    def logistic_probs(self, X, beta):
        logits = X @ beta
        return 1 / (1 + np.exp(-logits))

    def compute_logistic_gradient(self, X, y, beta):
        # X: [n_samples, n_selected_features]
        # beta: [n_selected_features]
        p = self.logistic_probs(X, beta)
        return X.T @ (y - p)

    def _test_func(proposed_func):
        pass