from typing import List, Optional
from numbers import Number
from NewLogRegpy.utilities.upper_bounding_func import UpperBounder
from NewLogRegpy.tests.test_logger import TestLogger
from NewLogRegpy.tree.node import Node
from NewLogRegpy.utilities.brancher import Brancher
import json
import math
import time
import traceback
import heapq
from multiprocessing import Process, Pipe
  

class ParallelTree:

    def __init__(
            self,
            n: int,
            k: int,
            brancher: Brancher,
            initial_upper_bound_strategy: UpperBounder = None,
            test_logger: TestLogger = None
            ) -> None:
        self.k: int = k
        self.n: int = n
        
        Node.configure(n=n,k=k)
        
        self.brancher: Brancher = brancher
        self.initial_upper_bounder = initial_upper_bound_strategy
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.unexplored_internal_nodes: List[Node] = [] # Used as a min-heap via the heapq module
        self.number_infeasible_nodes_explored: int = 0
        self.best_feasible_node: Node = None
        self.number_feasible_nodes_explored: int = 0
        
        ### Research Specific Objects ###
        self.known_fixed_in: Optional[List[int]] = None
        self.variable_scores: Optional[List[float]] = None

        self.test_logger = test_logger
        
        ### Framework Metrics ###
        self._status: Optional[str] = None
        self._value = None
        self._solution = None
        self.num_iter: int = 0
        self.UB_update_iterations: List[int] = []
        self.LB_update_iterations: List[int] = []
        self.solve_time: float = 0 # total enumeration time
        self.lb_bound_time: float = 0
        self.ub_bound_time: float = 0
        self.var_selection_time: float = 0
        self.initial_gap: float = math.inf
        self.initial_LB: float = -math.inf
        self.initial_UB: float = math.inf

        self.initial_tree_size = self._subtree_size(0,0)
        self.remaining_tree_size = self._subtree_size(0,0)

    @property
    def gap(self):
        if self.UB == math.inf:
            return math.inf
        elif self.UB == 0:
            return 0
        else:
            return (self.UB - self.LB) / self.UB
    
    @property
    def bound_time(self) -> float:
        return self.lb_bound_time + self.ub_bound_time
    
    def solve(self,
              brancher_count: int,
              eps: Number = 0.0001,
              timeout: Number = 60,
              fixed_in_vars: List[int] = None,
              fixed_out_vars: List[int] = None,
              max_iter = 10000,
              safe_close_file = None
              ) -> bool:
        """Enumerate a branch and bound tree to solve the logistic regression problem to global
        optimality using the bounding and objective functions passed into the tree upon its
        construction.

        Populates the :code:'status' and :code:'value' attributes on the
        tree object as a side-effect.

        Arguments
        ----------
        eps: positive float, optional
            The desired optimality tolerance.
            The default tolerance is 1e-8.
        timeout: float, optional
            The number of minutes solve will run before terminating.
            The default timeout is after 60 minutes.
        fixed_vars: List[int], optional
            Variable elements known to be fixed in (i.e. x[i] = 1 forall i in fixed_vars).
        branch_strategy: str, optional
            The method for choosing subproblems in the tree.
            Defaults to depth first search (dfs). Any other input
            will result in a lower bound shrinking strategy.
        
        Returns
        -------
        bool: Whether or not the problem was solved to global optimality.
        
        Raises
        ------
        AssertionError
            Raised if epsilon or timeout are not Numbers.
            Raised if var_scores doesn't contain enough scores.
        ValueError
            Raised if a fixed variable index is negative or >= n
        """
        
        ######### SETUP START #########
        
        start_time = time.time()
        loop_time = time.time() - start_time

        assert isinstance(eps, Number), "eps must be a Number"
        assert eps > 0, "eps must be positive."
        assert isinstance(timeout, Number), "timeout must be a Number"

        # keep track of these for metric purposes
        self.eps = eps
        self.timeout = timeout

        # check if there are fixed variables
        fixed_in_0, fixed_out_0 = 0, 0
        if fixed_in_vars != None:
            for i in range(len(fixed_in_vars)):
                assert isinstance(fixed_in_vars[i], int), "the fixed_in_vars list must contain integers."
            fixed_in_0 = Node.iter_to_varbitset(fixed_in_vars)
        if fixed_out_vars != None:
            for i in range(len(fixed_out_vars)):
                assert isinstance(fixed_out_vars[i], int), "the fixed_out_vars list must contain integers."
            fixed_out_0 = Node.iter_to_varbitset(fixed_out_vars)
        # Create root node
        root_node: Node = Node(fixed_in_0, fixed_out_0)
        if root_node.is_terminal_leaf():
            root_node.lb, _ = self.brancher.evaluate_single_node(root_node)
            self.UB = root_node.lb
            self.best_feasible_node = root_node
            self.number_feasible_nodes_explored += 1
            self.remaining_tree_size = 0
        else:
            root_node.lb, _ = self.brancher.evaluate_single_node(root_node)
            self.unexplored_internal_nodes.append(root_node)
            self.remaining_tree_size -= 1
        self.LB = root_node.lb

        # Try initial upper bounder
        if self.initial_upper_bounder != None:
            self.initial_UB, initial_ub_time, ub_fixed_in = self.initial_upper_bounder()
            self.UB = self.initial_UB
            initial_ub_node: Node = Node(ub_fixed_in, 0)
            initial_ub_node.lb = self.initial_UB
            self.best_feasible_node = initial_ub_node
            self.number_feasible_nodes_explored += 1
            self.ub_bound_time += initial_ub_time
            print("Checking initial upper bound is feasible:", initial_ub_node.is_terminal_leaf())
            
        # First round of logs
        print(f"Setup complete | UB = {self.UB:.4f} | gap = {self.gap:.4f} | Open Subproblems: {len(self.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {self.remaining_tree_size:,} | Running Time: {loop_time:.2f} seconds") 
        
        if self.test_logger != None:
            self.test_logger.log(0, time.time() - start_time, self.UB, self.LB, len(self.unexplored_internal_nodes), self.remaining_tree_size)

        brancher_lookup = []
        for i in range(brancher_count):
            parent_conn, child_conn = Pipe()
            p = Process(target=worker_loop, args=(self.brancher, child_conn, self.k, self.n))
            p.start()
            brancher_lookup.append([parent_conn, p, None])

        ######### SETUP END #########

        
        ######### MAIN #########
        
        print("Gap greater than epsilon:", self.gap > eps)
        print("Timeout greater than loop time:", timeout > (loop_time / 60))

        while (self.gap > eps and timeout > (loop_time / 60) and self.num_iter <= max_iter):
            if (self.gap > eps and len(self.unexplored_internal_nodes) == 0):
                raise Exception("Node list is empty but GAP is unsatisfactory.")
            UB_updated = False
            
            # Send nodes to free branchers
            for i in range(brancher_count):
                if brancher_lookup[i][2] is None:
                    if self.unexplored_internal_nodes:
                        node = heapq.heappop(self.unexplored_internal_nodes)
                        brancher_lookup[i][0].send(("SPLIT", node.to_dict()))
                        brancher_lookup[i][2] = node
                    else:
                        break
            
            # Attempt to collect branches from busy branchers
            for i in range(brancher_count):
                if brancher_lookup[i][0].poll():  # returns True if there's something to read
                    data = brancher_lookup[i][0].recv()
                    if data[0] == "ERROR":
                        print(data, "\n\n\n")
                    branch = [Node.from_dict(node) for node in data[1][0]]
                    brancher_lookup[i][2] = None
                    # Send back any available nodes to improve utilization
                    if self.unexplored_internal_nodes:
                        node = heapq.heappop(self.unexplored_internal_nodes)
                        brancher_lookup[i][0].send(("SPLIT", node.to_dict()))
                        brancher_lookup[i][2] = node
                    # Deal with branch
                    for node in branch:
                        if node.is_terminal_leaf:
                            self.remaining_tree_size -= 1
                            self.number_feasible_nodes_explored += 1
                            if node.lb < self.UB:
                                self.best_feasible_node = node
                                self.UB = node.lb
                                UB_updated = True
                        else:
                            if node.lb < self.UB:
                                heapq.heappush(self.unexplored_internal_nodes, node)
                                self.remaining_tree_size -= 1
                            else:
                                self.remaining_tree_size -= self._subtree_size(len(node.fixed_in), len(node.fixed_out))
                    self.num_iter += data[1][1]
                    self.remaining_tree_size -= data[1][2]
            
            # Pruning
            if UB_updated:
                unexplored_nodes = []
                for x in self.unexplored_internal_nodes:
                    if ((self.UB - x.lb)  / self.UB) > eps:
                        unexplored_nodes.append(x)
                    else:
                        self.remaining_tree_size -= self._subtree_size(len(x.fixed_in), len(x.fixed_out)) - 1
                heapq.heapify(unexplored_nodes)
                self.unexplored_internal_nodes = unexplored_nodes
            
            # Updating LB
            if self.unexplored_internal_nodes:
                temp_list = [self.unexplored_internal_nodes[0]] + [brancher_lookup[i][2] for i in range(brancher_count) if brancher_lookup[i][2] is not None]
            else:
                temp_list = [brancher_lookup[i][2] for i in range(brancher_count) if brancher_lookup[i][2] is not None]
            if temp_list:
                self.LB = min(temp_list).lb
            else:
                self.LB = self.UB

            loop_time = time.time() - start_time

            if (self.gap > eps and len(temp_list) == 0):
                raise Exception("Node list is empty but GAP is unsatisfactory.")
            
            print(f"\033[KIteration {self.num_iter} | UB = {self.UB:.4f} | gap = {self.gap:.4f} | Open Subproblems: {len(temp_list)}"
                + f" | Tree Remaining: {self.remaining_tree_size:,} | Running Time: {loop_time:.2f} seconds", end = "\r") 
            if self.test_logger != None:
                self.test_logger.log(self.num_iter, loop_time, self.UB, self.LB, len(temp_list), self.remaining_tree_size)

        print()
        for i in range(brancher_count):
            if brancher_lookup[i][0].poll():   # Only recv if something is waiting
                brancher_lookup[i][0].recv()
            brancher_lookup[i][0].send(("STOP", None))
            brancher_lookup[i][1].join()
            brancher_lookup[i][0].close()

        
        if (timeout < loop_time / 60) or self.num_iter > max_iter:            
            self._status = "solve timed out."
            print("\nSolve timed out. Runtime:", time.time() - start_time)
            if safe_close_file is not None:
                with open(safe_close_file, mode='w') as f:
                    json.dump({"UB": int(self.UB),
                                "UB node": self.best_feasible_node.to_dict(),
                                "iteration": int(self.num_iter),
                                "eps": float(eps),
                                "open_subproblems": [node.to_dict() for node in self.unexplored_internal_nodes]}, f)
            return False
        
        self._status = "global optimal found."
        self._value = self.UB
        self._solution = self.best_feasible_node
        self.solve_time = time.time() - start_time
        if self.test_logger != None:
                self.test_logger.log(self.num_iter, loop_time, self.UB, self.LB, len(self.unexplored_internal_nodes), self.remaining_tree_size)
        print("\nFound global optimal. Runtime:", self.solve_time)
        return True
    

    def _subtree_size(self, fixed_in_len, fixed_out_len):
        # Returns the size of the subtree starting at the root node specified
        return 2 * math.comb(self.n - fixed_in_len - fixed_out_len, self.k - fixed_in_len) - 1


# Message types we'll send/receive via queues
# send to worker: ("SPLIT", Node dictionary)
# shutdown: ("SHUTDOWN",)
# worker responds: ("DONE", (branch, num_iter, skipped_nodes)) or ("ERROR", exc_str)

def worker_loop(brancher, conn, k, n):
    """
    Each process runs this loop. We create the brancher here so heavy objects (phi/f0)
    are created once per worker process instead of pickled each task.
    """
    # Create persistent object in this process
    Node.configure(n=n,k=k)

    start_time = time.time()
    first_time = True
    event_time = time.time()
    utilized_time = 0

    while True:
        msg = conn.recv()
        if msg[0] == "SPLIT":
            if first_time:
                start_time = time.time()
                first_time = False
            event_time = time.time()
            try:
                current_node = Node.from_dict(msg[1])
                branch, num_iter, skipped_nodes = brancher.branch_node(current_node)
                branch = [node.to_dict() for node in branch]
                utilized_time += time.time() - event_time
                conn.send(("DONE", (branch, num_iter, skipped_nodes)))
            except Exception:
                conn.send(("ERROR", traceback.format_exc()))
        elif msg[0] == "STOP":
            print("utilization:", utilized_time / (time.time() - start_time))
            break
    conn.close()

