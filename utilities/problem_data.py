import numpy as np

class ProblemData():
    def __init__(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.n: int = X.shape[1]
        self.k: int = k
        assert k <= self.n, 'k must be less than or equal to the number of attributes'
        assert k >= 1, 'k must be positive'