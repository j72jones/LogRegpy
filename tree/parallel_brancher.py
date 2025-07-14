from LogRegpy.tree.node import Node
from copy import deepcopy



class parallelBrancher:
    def __init__(self, phi, f0):
        self.phi = phi
        self.f0 = f0


    def _split_problem(self, node: Node):
        """
        For visualization purposes, assume "left" subproblem corresponds to selecting
        a variable while a "right" subproblem corresponds to discarding a variable.

        Don't need to handle if node is leaf since all leaves are passed into
        feasible solutions only.

        when adding see if pruning should take place
        or if leaf node conditions hold.

        For a chosen variable there are six possible conditions to consider
        WLOG, for x variable
        1. 
        2. fixed_in_full -> create right problem (make a terminal leaf)
        3. fixed_out_full -> create left subproblem (make a terminal leaf)
        4. internal node -> create two subproblems

        Note that the following conditions are handled prior to the if statements:
        1. WLOG node is_x_terminal_leaf -> ______________
        2. node is a terminal leaf node -> doesn't get added to L_k to begin with.
        
        """
        chosen_var, var_choice_time = self.next_var(node)
        self.var_selection_time += var_choice_time

        if chosen_var is not None:
            left_node = self._create_left_subproblem(node, chosen_var)
            right_node = self._create_right_subproblem(node, chosen_var)
        else:
            raise Exception("Branching code ran into an unexpected case") # TODO: add information here, i.e. print something

        return left_node, right_node
    

    def _create_left_subproblem(self, node: Node, branch_idx: int) -> None:
        """
        fixes in:
        - adds the new index to fixed_in
        - creates corresponding node
        """
        new_fixed_in = deepcopy(node.fixed_in) + [branch_idx]
        new_subproblem: Node = Node(new_fixed_in, node.fixed_out)
        
        self._evaluate_node(new_subproblem, previous_node=node)

        return new_subproblem
    

    def _create_right_subproblem(self, node: Node, branch_idx: int) -> None:
        """
        fixes out:
        - adds the new index to fixed_out
        - creates corresponding node
        """
        new_fixed_out = deepcopy(node.fixed_out) + [branch_idx]
        new_subproblem: Node = Node(node.fixed_in, new_fixed_out)
        
        self._evaluate_node(new_subproblem)

        return new_subproblem


    def _evaluate_node(self, node: Node, previous_node=None) -> None:
        # note that if you want generalilzation of bounding/obj functions then they need
        # to return bound val, bounding time respectively.

        if node.is_terminal_leaf:
            node.lb, bound_time = self.f0(node)
        else:
            if previous_node is not None and self.phi_fixed_in_agnostic:
                node.lb = previous_node.lb
                node.coefs = previous_node.coefs
            else:
                node.lb, bound_time = self.phi(node)