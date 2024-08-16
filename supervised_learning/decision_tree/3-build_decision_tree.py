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

        while stack:
            element, is_leaf = stack.pop(0)

            if is_leaf:
                yield "->" + str(element)

            if element.left_child:
                stack.append(
                    (element.left_child, element.left_child.is_leaf)
                )
            if element.right_child:
                stack.append(
                    (element.right_child, element.right_child.is_leaf)
                )


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
