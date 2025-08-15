from enum import Enum
from typing import List, Optional
from numbers import Number
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from LogRegpy.tests.test_logger import TestLogger
from LogRegpy.tree.node import Node
from LogRegpy.tree.parallel_brancher import ParallelBrancher
from copy import deepcopy
from numpy import argmax, argmin
import math
import time
from numpy import (setdiff1d)
import traceback

# worker_pool.py
from multiprocessing import Process, Pipe


class BranchStrategy(Enum):
    SHRINK = 1
    DFS = 2
    

class ParallelTree:

    def __init__(
            self,
            n: int,
            k: int,
            obj: Objective,
            lower_bounder: Bounder,
            var_select: VariableChooser,
            lower_bounder_fixed_in_agnostic = False,
            branch_strategy: BranchStrategy = BranchStrategy.SHRINK,
            initial_upper_bound_strategy: UpperBounder = None,
            test_logger: TestLogger = None
            ) -> None:
        self.k: int = k
        self.n: int = n
        Node.k = k
        Node.n = n
        
        self.f0: Objective = obj
        self.phi: Bounder = lower_bounder
        self.phi_fixed_in_agnostic = lower_bounder_fixed_in_agnostic
        self.next_var = var_select
        self.initial_upper_bounder = initial_upper_bound_strategy
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.unexplored_internal_nodes: List[Node] = []
        self.number_infeasible_nodes_explored: int = 0
        self.best_feasible_node: Node = None
        self.number_feasible_nodes_explored: int = 0

        self.branch_strategy: BranchStrategy = branch_strategy
        
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
        if self.LB == -math.inf:
            return math.inf
        elif self.LB == 0:
            return (self.UB - self.LB)
        else:
            return (self.UB - self.LB) / self.LB
    
    @property
    def bound_time(self) -> float:
        return self.lb_bound_time + self.ub_bound_time
    
    def solve(self,
              brancher_count: int,
              eps: Number = 0.0001,
              timeout: Number = 60,
              fixed_in_vars: List[int] = None,
              fixed_out_vars: List[int] = None,
              #var_scores: List[float] = None,
              branch_strategy: str = "shrink",
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

        if self.initial_upper_bounder != None:
            self.initial_UB, initial_ub_time, ub_fixed_in = self.initial_upper_bounder()
            self.UB = self.initial_UB
            initial_ub_node: Node = Node(ub_fixed_in, [])
            initial_ub_node.lb = self.initial_UB
            self.best_feasible_node = initial_ub_node
            self.number_feasible_nodes_explored += 1
            self.ub_bound_time += initial_ub_time
            print("Checking initial upper bound is feasible:", initial_ub_node.is_terminal_leaf)
            

        # check if there are fixed variables
        # create root node, gap, LB accordingly
        fixed_in_0, fixed_out_0 = [], []
        if fixed_in_vars != None:
            for i in range(len(fixed_in_vars)):
                assert isinstance(fixed_in_vars[i], int), "the fixed_in_vars list must contain integers."
            fixed_in_0 = deepcopy(fixed_in_vars)

        if fixed_out_vars != None:
            for i in range(len(fixed_out_vars)):
                assert isinstance(fixed_out_vars[i], int), "the fixed_out_vars list must contain integers."
            fixed_out_0 = deepcopy(fixed_out_vars)

        self._create_root_node(fixed_in_0, fixed_out_0)

        print(f"Setup complete | UB = {self.UB:.4f} | gap = {self.gap:.4f} | Open Subproblems: {len(self.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {self.remaining_tree_size:,} | Running Time: {loop_time:.2f} seconds") 
        
        if self.test_logger != None:
            self.test_logger.log(0, time.time() - start_time, self.UB, self.LB, len(self.unexplored_internal_nodes), self.remaining_tree_size)

        brancher_lookup = []
        for i in range(brancher_count):
            parent_conn, child_conn = Pipe()
            p = Process(target=worker_loop, args=(self.f0, self.next_var,   child_conn, self.k, self.n))
            p.start()
            brancher_lookup.append([parent_conn, p, None])

        self.UB_updated = False

        ######### SETUP END #########

        
        ######### MAIN #########
        
        print("Gap greater than epsilon:", self.gap > eps)
        print("Timeout greater than loop time:", timeout > (loop_time / 60))

        while (self.gap > eps and timeout > (loop_time / 60) and self.num_iter <= max_iter):
            # Pruning
            if self.UB_updated:
                unexplored_nodes = []
                for x in self.unexplored_internal_nodes:
                    if ((self.UB - x.lb)  / x.lb) > eps:
                        unexplored_nodes.append(x)
                    else:
                        self.remaining_tree_size -= self._subtree_size(len(x.fixed_in), len(x.fixed_out)) - 1
                self.unexplored_internal_nodes = unexplored_nodes
            
            self.UB_updated = False
            
            for i in range(brancher_count):
                if brancher_lookup[i][2] is None:
                    if self.unexplored_internal_nodes:
                        node: Node = self._choose_subproblem()
                        brancher_lookup[i][0].send(("SPLIT", node.to_dict()))
                        brancher_lookup[i][2] = node
                    else:
                        break

            for i in range(brancher_count):
                if brancher_lookup[i][0].poll():  # returns True if there's something to read
                    
                    data = brancher_lookup[i][0].recv()
                    if data[0] == "ERROR":
                        print(data, "\n\n\n")
                    left_node, right_node = Node.from_dict(data[1][0]), Node.from_dict(data[1][1])
                    brancher_lookup[i][2] = None
                    self._manage_node(left_node)
                    self._manage_node(right_node)
                    self.num_iter += 1

            temp_list = self.unexplored_internal_nodes + [brancher_lookup[i][2] for i in range(brancher_count) if brancher_lookup[i][2] is not None]
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

        for i in range(brancher_count):
            brancher_lookup[i][0].send(("STOP", None))
            brancher_lookup[i][1].join()
            brancher_lookup[i][0].close()

        
        if (timeout < loop_time / 60) or self.num_iter > max_iter:            
            self._status = "solve timed out."
            print("\nSolve timed out. Runtime:", time.time() - start_time)
            if safe_close_file is not None:
                import json
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
    

    def _create_root_node(self, fixed_in, fixed_out):
        root_node: Node = Node(fixed_in, fixed_out)
        
        if root_node.is_terminal_leaf:
            root_node.lb, root_obj_time = self.phi(root_node)
            self.UB = root_node.lb
            self.ub_bound_time += root_obj_time
            self.UB_update_iterations.append(self.num_iter)
            self.best_feasible_node = root_node
            self.number_feasible_nodes_explored += 1
            self.remaining_tree_size = 0
        else:
            root_node.lb, root_obj_time = self.phi(root_node)
            self.unexplored_internal_nodes.append(root_node)

        self.LB = root_node.lb
        self.lb_bound_time += root_obj_time
        self.LB_update_iterations.append(self.num_iter)

        self.initial_gap = self.UB - self.LB

    
    def _choose_subproblem(self) -> Node:
        if self.branch_strategy == BranchStrategy.SHRINK:
            return self.unexplored_internal_nodes.pop(argmin(self.unexplored_internal_nodes))
        else:
            return self.unexplored_internal_nodes.pop()
        

    def _manage_node(self, node):
        if node.is_terminal_leaf:
            self.remaining_tree_size -= 1
            self.number_feasible_nodes_explored += 1
            if node.lb < self.UB:
                self.best_feasible_node = node
                self.UB = node.lb
                self.UB_updated = True
        else:
            if node.lb < self.UB:
                self.unexplored_internal_nodes.append(node)
                self.remaining_tree_size -= 1
            else:
                self.remaining_tree_size -= self._subtree_size(len(node.fixed_in), len(node.fixed_out))


    def _subtree_size(self, fixed_in_len, fixed_out_len):
        # Returns the size of the subtree starting at the root node specified
        return 2 * math.comb(self.n - fixed_in_len - fixed_out_len, self.k - fixed_in_len) - 1











# Message types we'll send/receive via queues
# send to worker: ("WORK", Node, maybe_prev_node)
# shutdown: ("SHUTDOWN",)
# worker responds: ("RESULT", worker_id, left_node, right_node) or ("ERROR", worker_id, exc_str)

def worker_loop(f0, next_var, conn, k, n):
    """
    Each process runs this loop. We create the brancher here so heavy objects (phi/f0)
    are created once per worker process instead of pickled each task.
    """
    # Create persistent object in this process
    Node.k = k
    Node.n = n
    brancher = ParallelBrancher(f0, next_var)

    while True:
        msg = conn.recv()
        if msg[0] == "SPLIT":
            try:
                current_node = Node.from_dict(msg[1])
                left_node, right_node = brancher._split_problem(current_node)
                conn.send(("DONE", (left_node.to_dict(), right_node.to_dict())))
            except Exception:
                conn.send(("ERROR", traceback.format_exc()))
        elif msg[0] == "STOP":
            break
    conn.close()

