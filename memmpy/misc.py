"""
Author:  Yakov Vaisbourd  2021-2022
"""

import numpy as np
import matplotlib.pyplot as plt


MAX_ITER = 200
TOLERANCE = 1e-12


# Auxiliary methods
def power_method(lin_map, lin_map_adj, dom_shape):
    x = np.random.rand(*dom_shape)
    for k in np.arange(MAX_ITER):
        x = x / np.linalg.norm(x, 'fro')
        x = lin_map_adj(lin_map(x))

    x = x / np.linalg.norm(x, 'fro')
    x = lin_map(x)

    return np.linalg.norm(x, 'fro')


def find_root(func_val_der, lower_bound, upper_bound, initial_point, max_iter=MAX_ITER, tol=TOLERANCE):
    # Safeguarded Newton-Raphson method under monotonicity assumption. If func_der is None then bisection.
    # Assumptions lb<=ub, tol >=0, f is monotonically increasing.

    x = np.copy(initial_point)
    lb = np.copy(lower_bound)
    ub = np.copy(upper_bound)
    x = np.maximum(x, lb)
    x = np.minimum(x, ub)

    index_not_converged = np.ones_like(x, dtype=bool)
    f_val = np.empty_like(x)
    f_der = np.empty_like(x)


    f_val[index_not_converged], f_der[index_not_converged] = func_val_der(x, index_not_converged)

    abs_f_der_geq_tol = np.abs(f_der) > TOLERANCE
    f_val_geq_tol = f_val > tol
    f_val_leq_tol = f_val < -tol
    index_not_converged = np.logical_or(f_val_geq_tol, f_val_leq_tol)

    ub[f_val_geq_tol] = x[f_val_geq_tol]
    lb[f_val_leq_tol] = x[f_val_leq_tol]

    iter = 0
    stopping_criteria_flag = index_not_converged.any()
    while stopping_criteria_flag:
        iter += 1
        if not np.any(np.isnan(f_der)):
            np.copyto(x, x - np.divide(f_val, f_der, out=np.zeros_like(f_val), where=abs_f_der_geq_tol),
                      where=index_not_converged)

            out_of_bounds = np.logical_or(x >= ub, x <= lb)

        else:
            out_of_bounds = index_not_converged

        np.copyto(x, (lb + ub) / 2, where=np.logical_or(out_of_bounds, np.logical_not(abs_f_der_geq_tol)))

        f_val[index_not_converged], f_der[index_not_converged] = func_val_der(x, index_not_converged)
        f_val_geq_tol = f_val > tol
        f_val_leq_tol = f_val < -tol
        index_not_converged = np.logical_or(f_val_geq_tol, f_val_leq_tol)
        abs_f_der_geq_tol = np.abs(f_der) > TOLERANCE

        ub[f_val_geq_tol] = x[f_val_geq_tol]
        lb[f_val_leq_tol] = x[f_val_leq_tol]

        if iter >= max_iter or not index_not_converged.any():
            stopping_criteria_flag = False

    return x

