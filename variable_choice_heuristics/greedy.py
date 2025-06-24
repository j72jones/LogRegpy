from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import math
import time

class Greedy(VariableChooser):
    def __init__(self, data: ProblemData) -> None:
        self.data = data

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        best_var = None
        min_objective = math.inf
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

            if min(objective_out, objective_in) < min_objective:
                best_var = i
                min_objective = min(objective_out, objective_in)
        return (best_var, start_time - time.time())

    def _test_func(proposed_func):
        pass


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
        node.model = LogisticRegression(**self.params)
        x = data.X[:, attributes]
        node.model.fit(x, data.y)

        # Calculate score
        return node.model.score(x, data.y), time.time() - start_time