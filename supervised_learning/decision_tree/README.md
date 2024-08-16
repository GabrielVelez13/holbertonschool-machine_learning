## README for Decision Trees

### Table of Contents
1. [Introduction](#introduction)
2. [Components](#components)
3. [Building a Decision Tree](#building-a-decision-tree)
4. [Example Usage](#example-usage)
5. [References](#references)

### Introduction
A decision tree is a flowchart-like structure used for decision-making and predictive modeling. Each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).

### Components
- **Node**: Represents a decision point or a test on an attribute.
- **Leaf**: Represents a class label or decision.
- **Decision Tree**: The entire structure starting from the root node to the leaf nodes.

### Building a Decision Tree
To build a decision tree, you need to:
1. Define the `Node` and `Leaf` classes.
2. Implement methods to split the data and create child nodes.
3. Define a `Decision_Tree` class to manage the tree structure and provide methods for training and prediction.

### Example Usage

#### Node Class
```python
class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Implementing a max depth algorithm using DFS iterative"""
        stack = [(self, self.depth)]
        max_depth = -float('inf')

        while stack:
            node, curr_depth = stack.pop()
            max_depth = max(max_depth, curr_depth)

            if node.left_child:
                stack.append((node.left_child, node.left_child.depth))
            if node.right_child:
                stack.append((node.right_child, node.right_child.depth))

        return max_depth
```

#### Leaf Class
```python
class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth
```

#### Decision Tree Class
```python
class Decision_Tree:
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
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
        return self.root.max_depth_below()
```

#### Main Script
```python
#!/usr/bin/env python3

Node = __import__('0-build_decision_tree').Node
Leaf = __import__('0-build_decision_tree').Leaf
Decision_Tree = __import__('0-build_decision_tree').Decision_Tree

def example_0():
    leaf0 = Leaf(0, depth=1)
    leaf1 = Leaf(0, depth=2)
    leaf2 = Leaf(1, depth=2)
    internal_node = Node(feature=1, threshold=30000, left_child=leaf1, right_child=leaf2, depth=1)
    root = Node(feature=0, threshold=.5, left_child=leaf0, right_child=internal_node, depth=0, is_root=True)
    return Decision_Tree(root=root)

def example_1(depth):
    level = [Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def get_v(node):
        if node.is_leaf:
            return node.value
        else:
            return node.threshold

    for d in range(depth):
        level = [Node(feature=0,
                      threshold=(get_v(level[2 * i]) + get_v(level[2 * i + 1])) / 2,
                      left_child=level[2 * i],
                      right_child=level[2 * i + 1], depth=depth - d - 1) for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return Decision_Tree(root=root)

print(example_0().depth())
print(example_1(5).depth())
```

### References
- [Wikipedia: Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)
- [Scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)