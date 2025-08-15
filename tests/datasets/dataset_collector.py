from enum import Enum
from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import make_classification
import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd

class uciDatasetsPy(Enum):
    SPECT_HEART = 95
    IONOSPHERE = 52
    WINE = 109
    MYOCARDIAL = 579
    MUSK = 74
    BANK_MARKETING = 222

uciFairDatasetsPy = {
    "ADULT": {
        "ID": 2,
        "FAIR CLASSIFIER": 'sex'
    }
}
    


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
        self.X: np.array = None
        self.y: np.array = None
        self.Q: np.array = None
        self.n: int = None
        self.rows: int = None
    
    def __call__(self, dataset_name: str) -> bool:
        if dataset_name == "BANK_MARKETING":
            # fetch dataset 
            dataset = fetch_ucirepo(id=uciDatasetsPy[dataset_name].value)
            # data (as pandas dataframes becomes np.ndarray) 
            self.X = dataset.data.features[["age", "default", "balance", "housing", "loan", "contact", "day_of_week", "month", "campaign", "pdays", "previous", "poutcome"]]
            self.X["contact"] = self.X["contact"] == "cellular"
            print(self.X["poutcome"])
            self.X["poutcome"] = self.X["poutcome"].apply(lambda x: -1 if pd.isna(x) else 0 if x == 'failure' else 1 if x == "nonexistent" else 2)
            self.X["default"] = self.X["default"] == "yes"
            self.X["housing"] = self.X["housing"] == "yes"
            self.X["loan"] = self.X["loan"] == "yes"
            self.X["day_of_week"] = self.X["day_of_week"] % 7
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            self.X["month"] = self.X["month"].str.lower().map(month_map)
            #print(self.X)
            self.X = self.X.fillna(0).to_numpy()
            self.y = dataset.data.targets.to_numpy().ravel()
            unique_vals = np.unique(self.y)
            if len(unique_vals) != 2:
                raise ValueError("Expected exactly two unique values")
            self.y = (self.y == unique_vals[1]).astype(int)
            self.rows, self.n = self.X.shape
        
        elif dataset_name in uciDatasetsPy.__members__:
            # fetch dataset 
            dataset = fetch_ucirepo(id=uciDatasetsPy[dataset_name].value)
            # data (as pandas dataframes becomes np.ndarray) 
            self.X = dataset.data.features.fillna(0).to_numpy()
            if dataset_name == "MYOCARDIAL":
                self.y = dataset.data.targets["ZSN"].to_numpy().ravel()
            else:
                self.y = dataset.data.targets.to_numpy().ravel()
            unique_vals = np.unique(self.y)
            if len(unique_vals) != 2:
                raise ValueError("Expected exactly two unique values")
            self.y = (self.y == unique_vals[1]).astype(int)
            self.rows, self.n = self.X.shape
            return True
        
        # if dataset_name in uciFairDatasetsPy.__members__:
        #     # fetch dataset 
        #     dataset = fetch_ucirepo(id=uciFairDatasetsPy[dataset_name].value)
        #     # data (as pandas dataframes becomes np.ndarray) 
        #     self.X = dataset.data.features.to_numpy()
        #     self.y = dataset.data.targets.to_numpy().ravel()
        #     unique_vals = np.unique(self.y)
        #     if len(unique_vals) != 2:
        #         raise ValueError("Expected exactly two unique values")
        #     self.y = (self.y == unique_vals[1]).astype(int)
        #     self.rows, self.n = self.X.shape
        #     return True
                
        elif dataset_name == "MADELON":
            self.X = np.loadtxt(uciDatasetsLocal[dataset_name]["data_file_path"])
            self.y = np.loadtxt(uciDatasetsLocal[dataset_name]["label_file_path"]).ravel()
            self.rows, self.n = self.X.shape
            return True
        
        elif dataset_name == "DOROTHEA":
            X_sparse = lil_matrix(uciDatasetsLocal[dataset_name]["samples, features"], dtype=np.uint8)
            with open(uciDatasetsLocal[dataset_name]["data_file_path"]) as f:
                for i, line in enumerate(f):
                    indices = list(map(int, line.strip().split()))
                    X_sparse[i, np.array(indices) - 1] = 1  # adjust for 1-based indexing
            # Convert to CSR for fast arithmetic and row slicing
            self.X = X_sparse.toarray()
            self.y = np.loadtxt(uciDatasetsLocal[dataset_name]["label_file_path"]).ravel()
            self.rows, self.n = self.X.shape
            return True
        
        elif dataset_name == 'SYNTHETIC':
            self.X, self.y = make_classification(n_samples=1000, n_features=5000, 
                           n_informative=50, n_redundant=100, random_state=42)
            self.rows, self.n = self.X.shape
            return True
        
        else:
            return False
        


if __name__ == "__main__":
    # # fetch dataset 
    # dataset = fetch_ucirepo(id=2)
    # # data (as pandas dataframes becomes np.ndarray) 
    # X = dataset.data.features
    # classifier = X['sex']
    # classifiers = classifier.unique()
    # indices = {classs:  classifier.index[classifier['sex'] == classs].tolist() for classs in classifiers}
    
    # X = X.to_numpy()
    # Q = None
    # y = dataset.data.targets.to_numpy().ravel()
    
    # print(classifier)

    dataset = fetch_ucirepo(id=579)
    print(dataset.data.targets.columns)