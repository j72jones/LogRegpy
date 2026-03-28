# type: ignore

from enum import Enum
from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import make_classification
import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd
from typing import Optional

class uciDatasetsPy(Enum):
    SPECT_HEART = 95
    IONOSPHERE = 52
    WINE = 109
    MYOCARDIAL = 579
    MUSK = 74
    BANK_MARKETING = 222



uciDatasetsLocal = {
    "DOROTHEA": {
        "data_file_path" : "LogRegpy/tests/datasets/dorothea/DOROTHEA/dorothea_train.data",
        "label_file_path" : "LogRegpy/tests/datasets/dorothea/DOROTHEA/dorothea_train.labels",
        "samples, features" : (800, 100000)
                 },
    "MADELON": {
        "data_file_path" : "LogRegpy/tests/datasets/madelon/MADELON/madelon_train.data",
        "label_file_path" : "LogRegpy/tests/datasets/madelon/MADELON/madelon_train.labels",
        "samples, features" : (500, 4400)
                 },
}


class DatasetCollector():
    def __init__(self):
        self.X: np.ndarray
        self.y: np.ndarray
        self.Q: np.ndarray
        self.n: int
        self.rows: int
    
    def __call__(self, dataset_name: str,
                 n_samples: int = 700,
                 n_features: int = 34,
                 n_informative: int = 5,
                 n_redundant: int = 22,
                 random_state: int = 42) -> bool:
      
        if dataset_name in uciDatasetsPy.__members__:
            # fetch dataset 
            dataset = fetch_ucirepo(id=uciDatasetsPy[dataset_name].value)
            # data (as pandas dataframes becomes np.ndarray) 
            self.X = dataset.data.features.fillna(0).to_numpy()
            if dataset_name == "MYOCARDIAL":
                self.y = dataset.data.targets["ZSN"].to_numpy().ravel()
            else:
                self.y = dataset.data.targets.to_numpy().ravel()
            X_mean = np.mean(self.X, axis=0, keepdims=True)
            X_std = np.std(self.X, axis=0, keepdims=True)
            X_std = np.where(X_std < 1e-8, 1.0, X_std)  # avoid divide-by-zero
            self.X = (self.X - X_mean) / X_std
            unique_vals = np.unique(self.y)
            if len(unique_vals) != 2:
                raise ValueError("Expected exactly two unique values")
            self.y = (self.y == unique_vals[1]).astype(int)
            self.rows, self.n = self.X.shape
            return True
                        
        elif dataset_name == "MADELON":
            self.X = np.loadtxt(uciDatasetsLocal[dataset_name]["data_file_path"])
            X_mean = np.mean(self.X, axis=0, keepdims=True)
            X_std = np.std(self.X, axis=0, keepdims=True)
            X_std = np.where(X_std < 1e-8, 1.0, X_std)  # avoid divide-by-zero
            self.X = (self.X - X_mean) / X_std
            self.y = np.loadtxt(uciDatasetsLocal[dataset_name]["label_file_path"]).ravel()
            unique_vals = np.unique(self.y)
            if len(unique_vals) != 2:
                raise ValueError("Expected exactly two unique values")
            self.y = (self.y == unique_vals[1]).astype(int)
            self.rows, self.n = self.X.shape
            return True
        
        elif dataset_name == "DOROTHEA":
            X_sparse = lil_matrix(uciDatasetsLocal[dataset_name]["samples, features"], dtype=np.uint8)
            with open(uciDatasetsLocal[dataset_name]["data_file_path"]) as f:
                for i, line in enumerate(f):
                    indices = list(map(int, line.strip().split()))
                    X_sparse[i, np.ndarray(indices) - 1] = 1  # adjust for 1-based indexing
            # Convert to CSR for fast arithmetic and row slicing
            self.X = X_sparse.toarray()
            self.y = np.loadtxt(uciDatasetsLocal[dataset_name]["label_file_path"]).ravel()
            self.rows, self.n = self.X.shape
            return True
        
        elif dataset_name == 'SYNTHETIC':
            self.X, self.y = make_classification(n_samples=n_samples, n_features=n_features, 
                           n_informative=n_informative, n_redundant=n_redundant, random_state=random_state)
            self.rows, self.n = self.X.shape
            return True
        
        return False