from multiprocessing import Lock
from sortedcontainers import SortedList
import math
from LogRegpy.tree.node import Node

class FeasibleSolutionHandler:
    def __init__(self):
        self.infeasible_queue = SortedList(key=lambda x: x.lb)  # sorted by lower bound
        self.nodes_being_explored = set()
        self.LB = -math.inf
        self.UB = math.inf
        self.best_feasible_node = None
        self.lock = Lock()

    def consider_feasible_node(self, node: Node, obj: float) -> bool:
        """
        If value improves the best value, keep it.
        """
        with self.lock:
            if obj < self.best_node.value:
                self.best_node.value = obj
                self.best_node.node = node
                # Prune
                i = self.queue.bisect_key_right(obj) # O(logn)
                queue = queue[:i]  # O(i)
                return True
            return False
        
    def get_best_feasible_node(self):
        with self.lock:
            return self.best_node.obj, self.best_node.value
        
    def get_UB(self):
        with self.lock:
            return self.best_feasible_node.value
        

    def add_infeasible_node(self, node, obj):
        with self.lock:
            if obj < self.best_feasible_node.value:
                self.queue.add(node)  # O(log n)
                self.LB = self.queue[0]
                return True
            return False

    def get_next_infeasible_node(self, current_node):
        with self.lock:
            self.nodes_being_explored.remove(current_node)
            if not self.queue:
                return None
            new_node = self.queue.pop(0) # O(1)
            self.nodes_being_explored.add(new_node)
            return new_node  


    def infeasible_queue_empty(self):
        with self.lock:
            return len(self.queue) == 0
    
    def infeasible_queue_length(self):
        with self.lock:
            return len(self.queue)
