from enum import Enum
from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd



class uciDatasets(Enum):
    SPECT_HEART = 95
    IONOSPHERE = 52
    WINE = 109


class DatasetCollector():
    def __init__(self):
        self.X: np.array = None
        self.y: np.array = None
        self.n: int = None
        self.rows: int = None
    
    def __call__(self, dataset_name: str) -> bool:
        if dataset_name in uciDatasets.__members__:
            # fetch dataset 
            dataset = fetch_ucirepo(id=uciDatasets[dataset_name].value)
            # data (as pandas dataframes becomes np.ndarray) 
            self.X = dataset.data.features.to_numpy()
            self.y = dataset.data.targets.to_numpy().ravel()
            self.rows, self.n = self.X.shape
            return True
        elif dataset_name == 'SYNTHETIC':
            self.X, self.y = make_classification(n_samples=1000, n_features=5000, 
                           n_informative=50, n_redundant=100, random_state=42)
            self.rows, self.n = self.X.shape
            return True
        else:
            return False
