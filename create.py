"""Create csv data for modelling.

Creates random csv data suitable for practicing
modelling.
"""

import random
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
import tree as tr


def head(a):
    """Extract first element of list.

    Input:
        a: any list
    Output:
        first element of list
    """
    if a == []:
        return []
    else:
        return a[0]


def tail(a):
    """Extract all but first element of list.

    Input:
        a: any list
    Output:
        all bar first element of list
    """
    if a == []:
        return []
    else:
        return a[1:]


def create_variable(v):
    """Create variable dictionary.

    Input:
        v: list of variable values
    Output:
        Dictionary of key-value pairs for variable
    """
    variable_dict = {}
    for i, entry in enumerate(v):
        variable_dict[i] = entry
    return variable_dict


def create_data_dictionary(h, v):
    """Create data dictionary.

    Inputs:
        h: list of column headers
        v: list of lists of values in same order as h
    Output:
        Ordered Dictionary of headers each entry being the
        key - value pairs for each variable
    """
    data_dict = OrderedDict()
    for i, entry in enumerate(h):
        data_dict[entry] = create_variable(v[i])
    return data_dict


def create_data(d, r):
    """Create sample data.

    Inputs:
    d: data dictionary for the dataset
    r: number of rows of data required
    Output:
    2 element list:
    1st element: length r list of classification
    2st element: length r list of features
    using dictionary d
    """
    data_features = list()
    data_class = list()
    for row in xrange(r):
        features = list()
        for entry in d:
            features.append(
                random.randint(0, max(d[entry].keys()))
            )
        data_class.append(head(features))
        data_features.append(tail(features))
    return [data_class, data_features]


def main():
    """Main program."""
    n_cases = 10
    headers = ["category", "sex", "age", "region", "health"]
    types = ['o','o','o','o','o']
    values = [
        ["success", "fail"],
        ["male", "female"],
        [x for x in xrange(16, 100)],
        ["north", "south", "east", "west"],
        ["good", "average", "poor"]
    ]

    data_dict = create_data_dictionary(headers, values)
    sample_data = create_data(data_dict, n_cases)
    ft = sample_data[1]
    print ft

    """
    enc = OneHotEncoder()
    categorical = [[x[0], x[2], x[3]] for x in ft]
    enc.fit(categorical)
    print enc.transform(categorical).toarray()
    """

    sample_tree = tr.create_simple_tree(sample_data[1], sample_data[0])
    node_values = tr.calculate_node_values(sample_tree, sample_data[1])
    training_tree = tr.build_decision_tree(
        sample_tree,
        node_values,
        data_dict
    )

    print training_tree


if __name__ == "__main__":
    main()
