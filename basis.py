"""Create set of binary combinations of categorical variable."""

import itertools


def invert_list(a):
    """Invert a binary list.

    Input:
        a: any list containing 1/0 values
    Output:
        a list with the 1/0 values inverted
    """
    inverted_list = [(not x) * 1 for x in a]
    return inverted_list


def create_true_false(a):
    """docstring."""
    i = 0
    true_false = list()
    len_a = len(a)
    while i <= len_a/2:
        true_false.append([a[i], a[len_a - i - 1]])
        i += 1
    return true_false


def combine_vectors(b, n):
    """Combine basis vectors.

    Calculates the unique set of linear combinations of length n
    from basis vectors b

    Inputs:
        b: set of basis vectors
        n: no of vectors to combine
    Outputs:
        unique combinations of basis vectors
    """
    vector_set = list()
    combs = itertools.combinations(b, n)
    for i in combs:
        comb_list = list()
        for j in i:
            comb_list.append(j)

        combined_list = list()
        for k in xrange(len(comb_list[0])):
            add_element = 0
            for g in comb_list:
                add_element += g[k]
            combined_list.append(add_element)
        vector_set.append(combined_list)
    return vector_set


def lin_comb_basis(b):
    """Create linear combination.

    Create the linear combinations of a list of basis vectors
    excluding (0,..,0) and (1,...,1)

    Input:
        b: list of basis vectors
    Output:
        a list of unique combinations
    """
    true_values = list()
    for k in xrange(1, len(b)):
        for j in combine_vectors(b, k):
            true_values.append(j)
    return true_values


def create_cat_splits(b):
    """Create categorical split variables.

    Creates list of true/false values for possible combinations of
    a categorical variable.

    Input:
        a: list of the categorical variable values
    Output:
        a list of length 2 lists containg the combinations
    """
    vectors = lin_comb_basis(b)
    return create_true_false(vectors)


def main():
    """Main function.

    Demo of how create_cat_splits works
    """
    basis = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]

    basis2 = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]

    for i in create_cat_splits(basis):
        print i

    for i in create_cat_splits(basis2):
        print i

    return True


if __name__ == "__main__":
    main()
