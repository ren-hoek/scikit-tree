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


def create_false_true(a):
    """Create a list of false.

    Pair up the categories to produce mutually exclusive combinations

    Input:
        a: single list of all potential category combinations
    Output:
        paired up list of the mutually exclusive category combinations
        (every value must belong to one of the 2 elements)
    """
    i = 0
    false_true = list()
    len_a = len(a)
    while i <= len_a/2:
        false_true.append([a[len_a - i - 1], a[i]])
        i += 1
    return false_true


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
        a list of length 2 lists containng the combinations
    """
    vectors = lin_comb_basis(b)
    return create_false_true(vectors)


def create_basis(n):
    """Create basis vectors.

    Create a list of n-dimensional basis vectors

    Input:
        n: number of dimensions
    Output:
        list of basis vectors
    """
    dim = xrange(n)
    basis = [[(j == i) * 1 for j in dim] for i in dim]
    return basis


def convert_to_cat(v, b):
    """Convert numerical category.

    Inputs:
        v: the value of the categorical variable
        b: the set of basis false/true vectors
    Output:
        a list of the basis false/true vectors v belongs to
    """
    return [(x[1][v] == 1) * 1 for x in b]


def create_categories(n):
    """Create mutually exclusive categorical features.

    Creates a list of the mutually exclusive categories
    using a basis of dimension n

    Inputs:
        n: dimensionality of the basis
    Output:
        a list of length 2 lists containng the combinations
    """
    return create_cat_splits(create_basis(n))


def create_cat_features(v, n):
    """Create categorical features.

    Creates a list of the categorical features for each
    element of v

    Inputs:
        v: the data as single column numerical list
        n: no of categories in v
    Outputs:
        a list containing the list of the binary features the element
        belongs to
    """
    return [convert_to_cat(x, create_categories(n)) for x in v]


def main():
    """Main function.

    Demo of how create_cat_features works
    """
    basis_4 = create_basis(4)

    basis_5 = create_basis(5)

    for i in create_cat_splits(basis_4):
        print i

    for i in create_cat_splits(basis_5):
        print i

    print convert_to_cat(2, create_categories(3))
    print convert_to_cat(1, create_categories(4))

    print create_categories(3)
    print create_cat_features([0, 1], 2)

    return True


if __name__ == "__main__":
    main()
