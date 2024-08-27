#!/usr/bin/env python3
""" Decision tree """
import numpy as np


class Node:
    """ Node class representing a decision point in the tree """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """ Initialize a Node with feature, threshold, children,
        and depth """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Implementing a max depth algorithm using DFS iterative """
        stack = [(self, self.depth)]
        max_depth = -float('inf')

        while stack:
            node, curr_depth = stack.pop()
            max_depth = max(max_depth, curr_depth)

            if node.left_child:
                stack.append(
                    (node.left_child, node.left_child.depth)
                )
            if node.right_child:
                stack.append(
                    (node.right_child, node.right_child.depth))

        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """ Count the number of nodes below this node """
        stack = [(self, self.is_leaf)]
        node = 0
        leaf = 0

        while stack:
            element, is_leaf = stack.pop()

            if is_leaf:
                leaf += 1
                node += 1
            else:
                node += 1

            if element.left_child:
                stack.append(
                    (element.left_child, element.left_child.is_leaf)
                )
            if element.right_child:
                stack.append(
                    (element.right_child, element.right_child.is_leaf)
                )

        return leaf if only_leaves else node

    def left_child_add_prefix(self, text):
        """ Add prefix to the left child text """
        lines = text.split("\n")
        new_text = "    +--->" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ Add prefix to the right child text """
        lines = text.split("\n")
        new_text = "    +--->" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """ String representation of the node """
        if self.is_leaf:
            return str(self)

        left = self.left_child_add_prefix(str(self.left_child)) \
            if self.left_child else ""
        right = self.right_child_add_prefix(str(self.right_child)) \
            if self.right_child else ""

        root = (f"root [feature={self.feature}, "
                f"threshold={self.threshold}]\n{left}{right}")
        node = (f" node [feature={self.feature}, "
                f"threshold={self.threshold}]\n{left}{right}").rstrip()
        if self.is_root:
            return root
        return node

    def get_leaves_below(self):
        """ Generator to yield leaves below this node """
        stack = [(self, self.is_leaf)]
        leaves = []
        while stack:
            element, is_leaf = stack.pop(0)

            if is_leaf:
                leaves.append(element)

            if element.left_child:
                stack.append(
                    (element.left_child, element.left_child.is_leaf)
                )
            if element.right_child:
                stack.append(
                    (element.right_child, element.right_child.is_leaf)
                )

        return leaves

    def update_bounds_below(self):
        """ Updates all upper and lower bounds """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

            if child == self.left_child:
                child.lower[self.feature] = self.threshold

            if child == self.right_child:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Consider the indicator function for a given node, denoted as
        “n.” This function is defined as follows:
        """
        def is_large_enough(x):
            lower = np.array([self.lower.get(i, -np.inf)
                              for i in range(x.shape[1])])
            return np.all(x > lower, axis=1)

        def is_small_enough(x):
            upper = np.array([self.upper.get(i, np.inf)
                              for i in range(x.shape[1])])
            return np.all(x <= upper, axis=1)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """ Predict the value for a given input x """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """ Leaf class representing a decision outcome in the tree """
    def __init__(self, value, depth=None):
        """ Initialize a Leaf with a value and depth """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Return the depth of the leaf """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Return the count of nodes below the leaf """
        return 1

    def __str__(self):
        """ String representation of the leaf """
        return f" leaf [value={self.value}]"

    def get_leaves_below(self):
        """ Return the leaf itself as it has no children """
        return [self]

    def update_bounds_below(self):
        """ pass because there is nothing below """
        pass

    def pred(self, x):
        """ Predict the value for a given input x """
        return self.value


class Decision_Tree:
    """ Decision_Tree class for managing the tree structure """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ Initialize a Decision_Tree with parameters
        and root node """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ Return the maximum depth of the tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Count the number of nodes in the tree """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ String representation of the decision tree """
        return self.root.__str__()

    def get_leaves(self):
        """ Get all leaves in the tree """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Updates the upper and lower bounds """
        self.root.update_bounds_below()

    def update_predict(self):
        """ Update the prediction function for the tree """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: sum(leaf.indicator(A)*leaf.value
                                     for leaf in leaves)

    def pred(self, x):
        """ Predict the value for a given input x """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target,
                                                dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory, 
                                                 self.target)}""")

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (self.explanatory[:, node.feature] <= node.threshold)

        right_population = node.sub_population & (self.explanatory[:, node.feature] > node.threshold)


        # Is left node a leaf ?
        is_left_leaf = node.is_leaf

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node,
                                                  left_population)
        else:
            node.left_child = self.get_node_child(node,
                                                  left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = node.is_leaf

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node,
                                                   right_population)
        else:
            node.right_child = self.get_node_child(node,
                                                   right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        value = node.value
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size