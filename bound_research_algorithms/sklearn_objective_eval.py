from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import time
from numpy import setdiff1d

def sklearn_objective_eval(data: ProblemData, node: Node) -> Tuple[float, float]:
    start_time = time.time()
    attributes = setdiff1d(range(data.n), node.fixed_out)
    # Train the logistic regression model
    node.model = LogisticRegression(penalty = None)
    x = data.X[:, attributes]
    node.model.fit(x, data.y)

    # Pull the probabilities
    y_prob = node.model.predict_proba(x)[:, 1]

    # Calculate log_loss (no regularization currently)
    return log_loss(data.y, y_prob), time.time() - start_time