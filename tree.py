"""Scikit Binary Decision Tree.

Python version:
    2.7
Required packages:
    scikit-learn (pip install scikit-learn)
    numpy (should be installed along with scikit-learn)

Functions for training and displaying a scikit-learn binary decsion tree

based on:
http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
"""

import numpy as np
from collections import Counter

from sklearn.datasets import load_iris
from sklearn import tree


def calculate_node_values(dec_tree, sample):
    """Construct counts of cases at each node.

    Inputs:
    dec_tree = the trained decision tree
    sample = the cases to pass through the tree

    Outputs:
    An dictionary with key of node_id and value of the
    node case count based on provided sample
    e.g. node_values[0] = 150
    150 cases in node[0] (the root node)
    """
    # the method .decision_path calculate the path through
    # tree for every case in sample
    node_indicator = dec_tree.decision_path(sample)

    # the method .apply calculates the node id of the
    # leaf node (terminal node) for each case in sample
    # It's commented out as it is not used in this function
    # leave_id = dt_iris.apply(iris.data)

    node_sum = []
    for j in xrange(len(sample)):
        node_index = node_indicator.indices[
            node_indicator.indptr[j]:
            node_indicator.indptr[j + 1]
        ]

        for k in node_index:
            node_sum.append(k)

    node_values = Counter(node_sum)
    return node_values


def build_decision_tree(dec_tree, node_values):
    """
    Build a string decision tree.

    Inputs:
    dec_tree = the trained decision tree
    node_values = the node counts for a particular sample

    Output:
    Returns a string representation of the decision tree
    with counts at each node based on values supplied in
    node_values
    """
    n_nodes = dec_tree.tree_.node_count
    children_left = dec_tree.tree_.children_left
    children_right = dec_tree.tree_.children_right
    feature = dec_tree.tree_.feature
    threshold = dec_tree.tree_.threshold

    node_depth = np.zeros(shape=n_nodes)

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    tree_output = ''
    tree_output += (
        "The binary tree structure has %s nodes and has "
        "the following tree structure:\n" % n_nodes
    )

    for i in range(n_nodes):
        if is_leaves[i]:
            tree_output += (
                "%snode=%s leaf node (%s).\n"
                % (int(node_depth[i]) * "\t",
                    i,
                    node_values[i])
            )
        else:
            tree_output += (
                "%snode=%s test node (%s): go to node %s if X[:, %s] "
                "<= %ss else to "
                "node %s.\n"
                % (int(node_depth[i]) * "\t",
                    i,
                    node_values[i],
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                   )
                )

    return tree_output


def run_decision_tree():
    """Main function.

    First runs a simple usage example.
    Then runs and display a tree based on example data
    from scikit-learn
    """
    """
    Simple usage example.

    X contains the featureset (also called independent variables)
    made from a list of n elements one for each case
    each element contains a list with the featureset for the case
    e.g. X contains 3 cases, each featureset contains 3 variables
    Y contains the classification variable (i.e. what you want to model)
    contains a single list with the class for each case
    e.g. 0 = Bad, 1 = Good or whatever the classification is
    """

    X = [[1, 1, 1], [1, 0, 0], [0, 0, 1]]
    Y = [1, 0, 1]

    # initialize the decision tree
    # various options can be specified here
    #   algorithm
    #   max number of leaf nodes etc
    dt = tree.DecisionTreeClassifier()

    # train the model (all scikit models follow the same data structure)
    # (can use other functions in scikit-learn to ceate train/test sets)
    # once trained can be used to predict new cases
    dt.fit(X, Y)

    # .predict predicts the class of new cases
    print "predicted "
    print dt.predict([[0, 0, 0], [2, 2, 2]])

    # .predict_proba gives the prob of the 2 classes
    print dt.predict_proba([[0, 0, 0], [2, 2, 2]])

    """
    Real data case.

    iris is an example dataset from scikit-learn
    """

    # load the iris data
    iris = load_iris()

    # uncomment these to see what the data looks like
    # print iris.data[:10]
    # print iris.target

    # initialize and train the decidion tree.
    # again in real life would split into train/test
    dt_iris = tree.DecisionTreeClassifier()
    dt_iris = dt_iris.fit(iris.data, iris.target)

    # calculate the node values based in iris.data
    node_values = calculate_node_values(dt_iris, iris.data)

    # produce decision tree with counts bases on iris.data
    output_tree = build_decision_tree(dt_iris, node_values)

    # print the tree to screen
    print output_tree

    return True


if __name__ == "__main__":
    run_decision_tree()
