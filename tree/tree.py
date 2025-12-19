from typing import List, Optional
from numbers import Number
from LogRegpy.utilities.brancher import Brancher
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from LogRegpy.tests.test_logger import TestLogger
from LogRegpy.tree.node import Node
import heapq
import math
import time

  

class Tree:

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
        
        self.brancher = brancher
        self.initial_upper_bounder = initial_upper_bound_strategy
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.unexplored_internal_nodes: List[Node] = []
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
            root_node.lb = self.brancher.evaluate_single_node(root_node)
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

        ######### SETUP END #########

        
        ######### MAIN #########
        
        print("Gap greater than epsilon:", self.gap > eps)
        print("Timeout greater than loop time:", timeout > (loop_time / 60))

        while (self.gap > eps and timeout > (loop_time / 60) and self.num_iter <= max_iter):
            if (self.gap > eps and len(self.unexplored_internal_nodes) == 0):
                raise Exception("Node list is empty but GAP is unsatisfactory.")
            UB_updated = False

            # Branch on node
            node: Node = heapq.heappop(self.unexplored_internal_nodes)
            branch, num_iter, skipped_nodes = self.brancher.branch_node(node)
            self.num_iter += num_iter
            self.remaining_tree_size -= skipped_nodes
            # Deal with branch
            for node in branch:
                if node.is_terminal_leaf():
                    self.remaining_tree_size -= 1
                    self.number_feasible_nodes_explored += 1
                    if node.lb < self.UB:
                        self.best_feasible_node = node
                        self.UB = node.lb
                        self.UB_updated = True
                else:
                    if node.lb < self.UB:
                        heapq.heappush(self.unexplored_internal_nodes, node)
                        self.remaining_tree_size -= 1
                    else:
                        self.remaining_tree_size -= self._subtree_size(node.len_fixed_in, node.len_fixed_out)

            # Pruning
            if UB_updated:
                unexplored_nodes = []
                for x in self.unexplored_internal_nodes:
                    if ((self.UB - x.lb)  / self.UB) > eps:
                        unexplored_nodes.append(x)
                    else:
                        self.remaining_tree_size -= self._subtree_size(x.len_fixed_in, x.len_fixed_out) - 1
                heapq.heapify(unexplored_nodes)
                self.unexplored_internal_nodes = unexplored_nodes

            # Updating LB
            if self.unexplored_internal_nodes:
                self.LB = self.unexplored_internal_nodes[0].lb
            else:
                self.LB = self.UB

            # Log current state
            loop_time = time.time() - start_time
            print(f"\033[KIteration {self.num_iter} | UB = {self.UB:.4f} | gap = {self.gap:.4f} | Open Subproblems: {len(self.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {self.remaining_tree_size:,} | Running Time: {loop_time:.2f} seconds", end = "\r") 
            if self.test_logger != None:
                self.test_logger.log(self.num_iter, loop_time, self.UB, self.LB, len(self.unexplored_internal_nodes), self.remaining_tree_size)
        
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
    
    def _subtree_size(self, fixed_in_len, fixed_out_len):
        # Returns the size of the subtree starting at the root node specified
        return 2 * math.comb(self.n - fixed_in_len - fixed_out_len, self.k - fixed_in_len) - 1