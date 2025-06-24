from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.variable_chooser import VariableChooser
from sklearn.linear_model import LogisticRegression
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import math
import time

class MaximumBoundTightening(VariableChooser):
    def __init__(self, data: ProblemData) -> None:
        self.data = data

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        best_var = None
        max_min_objective = -math.inf
        for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
            # First for fixing out
            attributes = setdiff1d(range(self.data.n), node.fixed_out + [i])
            # Train the logistic regression model
            model = LogisticRegression()
            x = self.data.X[:, attributes]
            model.fit(x, self.data.y)
            # Calculate scores
            objective_out = model.score(x, self.data.y)

            # Then for fixing in
            attributes = setdiff1d(range(self.data.n), node.fixed_out)
            # Train the logistic regression model
            model = LogisticRegression()
            x = self.data.X[:, attributes]
            model.fit(x, self.data.y)
            # Calculate scores
            objective_in = model.score(x, self.data.y)

            if min(objective_out, objective_in) > max_min_objective:
                best_var = i
                max_min_objective = min(objective_out, objective_in)
        return (best_var, start_time - time.time())

    def _test_func(proposed_func):
        pass