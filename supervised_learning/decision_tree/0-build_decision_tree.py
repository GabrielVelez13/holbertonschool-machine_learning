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
                    (node.right_child, node.right_child.depth)
                )

        return max_depth


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
