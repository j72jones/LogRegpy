import csv
import os

class TestLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = ["iteration", "time", "UB", "LB", "num_subproblems", "remaining_tree_size"]

        # If file doesn't exist, write header
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log(self, iteration: int, time: float, ub: float, lb: float, num_subproblems: int, remaining_tree_size: int):
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, time, ub, lb, num_subproblems, remaining_tree_size])

    def rewrite_file(self):
        # Rewrite with header
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)