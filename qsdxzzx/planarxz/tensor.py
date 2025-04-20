import numpy as np

# Optimized delta tensor


def delta(shape):
    """
    Return delta tensor of given shape, i.e. element values of 1 when all non-dummy indices are equal, 0 otherwise.

    :param shape: Shape of tensor.
    :type shape: tuple of int
    :return: Delta tensor
    :rtype: numpy.array
    """
    # non-dummy index mask


    ndi_mask = np.array(shape) != 1

    if any(ndi_mask):
    # start with ones, in case all dummy indices
        n = np.zeros(shape, dtype=int)
        index_arr = np.array([*np.ndindex(shape)])
        # Remove dummy indices
        trunc_arr = index_arr[:, ndi_mask]

        # Select rows with the same indices
        mask = np.all(trunc_arr[:, 1:] == trunc_arr[:, :-1], axis=1)
        selected_rows = index_arr[mask]

        # Change entries with the same indices to 1
        for lin in selected_rows:
            n[tuple(lin)] = 1


        return n
    else: 
        return np.ones(shape, dtype=int)