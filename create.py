"""Create csv data for modelling.

Creates random csv data suitable for practicing
modelling.
"""

import random
from collections import OrderedDict
# from sklearn.preprocessing import OneHotEncoder
import tree as tr
import basis as bs


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
    variable_dict = dict()
    for i, entry in enumerate(v):
        variable_dict[i] = entry
    return variable_dict


def create_data_dictionary(h, v, t):
    """Create data dictionary.

    Inputs:
        h: list of column headers
        v: list of lists of values in same order as h
        t: list of variable types in same order as h:
            'o' ordinal,
            'c' categorical
    Output:
        Ordered Dictionary of headers each entry being the
        key - value pairs for each variable
    """
    data_dict = OrderedDict()
    for i, entry in enumerate(h):
        data_dict[entry] = dict()
        data_dict[entry]["type"] = t[i]
        data_dict[entry]["length"] = len(v[i])
        data_dict[entry]["values"] = create_variable(v[i])
        if t[i] == 'c':
            data_dict[entry]["categories"] = bs.create_categories(len(v[i]))
        else:
            data_dict[entry]["categories"] = []

    return data_dict


def convert_value(v, d):
    """docstring."""
    if d == []:
        return [v]
    else:
        return [x for x in bs.convert_to_cat(v, d)]


def convert_dictionary(d):
    """Convert dictionary to binary.

    Converts the data dictionary to contain a set of
    binary variables for each categorical variable in
    the datset
    Input:
        d: dictionary with categorical single variables
    Output:
        dictionary with categorical variable as a set of binary
        variables
    """
    cat_dict = OrderedDict()
    k = 0
    for entry in d:
        if d[entry]["categories"]:
            for x in d[entry]["categories"]:
                cat_dict[k] = dict()
                cat_dict[k]['header'] = entry
                cat_dict[k]['type'] = 'c'
                cat_dict[k]['values'] = dict()
                for i, y in enumerate(x):
                    cat_dict[k]['values'][i] = ' '.join(
                        [(z * d[entry]["values"][j]) for j, z in enumerate(y)]
                    ).strip()
                k += 1
        else:
            cat_dict[k] = dict()
            cat_dict[k]['header'] = entry
            cat_dict[k]['type'] = 'o'
            cat_dict[k]['values'] = d[entry]["values"]

            k += 1
    return cat_dict


def create_data(d, r):
    """Create sample data.

    Inputs:
        d: data dictionary for the dataset
        r: number of rows of data required
    Output:
        2 element list using dictionary d:
            1st element: length r list of classification
            2st element: length r list of features
    """
    data_features = list()
    data_class = list()
    for row in xrange(r):
        features = list()
        for entry in d:
            value = random.randint(0, max(d[entry]["values"].keys()))
            features.append(value)
        data_class.append(head(features))
        data_features.append(tail(features))
    return [data_class, data_features]


def main():
    """Main program."""
    n_cases = 100
    headers = ["category", "sex", "age", "region", "health"]
    types = ['o', 'c', 'o', 'c', 'c']
    values = [
        ["success", "fail"],
        ["male", "female"],
        [x for x in xrange(16, 100)],
        ["north", "south", "east", "west"],
        ["good", "average", "poor"]
    ]

    data_dict = create_data_dictionary(headers, values, types)
    cat_data_dict = convert_dictionary(data_dict)
    sample_data = create_data(cat_data_dict, n_cases)

    sample_tree = tr.create_simple_tree(sample_data[1], sample_data[0])
    node_values = tr.calculate_node_values(sample_tree, sample_data[1])
    training_tree = tr.build_decision_tree(
        sample_tree,
        node_values,
        cat_data_dict
    )

    print training_tree


if __name__ == "__main__":
    main()
