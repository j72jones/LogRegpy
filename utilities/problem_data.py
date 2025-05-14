import numpy as np

class ProblemData():
    def __init__(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.n: int = X.shape[1]
        self.k: int = k
        assert k <= self.n, 'k must be less than the number of attributes'