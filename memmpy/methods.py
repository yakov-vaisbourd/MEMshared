"""
Author:  Yakov Vaisbourd  2021-2022
"""

import numpy as np
from memmpy.misc import power_method
from typing import NamedTuple


# Function Classes
class MathFunction:
    def __init__(self, val, grad=None):
        self.val = val
        self.grad = grad


class KernelFunction(MathFunction):
    def __init__(self, val, grad, grad_dual):
        super().__init__(val, grad)
        self.grad_dual = grad_dual

    def val_bregman(self, y, x):
        return self.val(y) - self.val(x) - np.dot(self.grad(x), y - x)


class ObjectiveFunction(MathFunction):
    def __init__(self, val, grad=None, residual=None, kernel=None, smoothness_constant=None, proximal_operator=None):
        super().__init__(val, grad)
        self.residual = residual
        self.kernel = kernel
        self.smoothness_constant = smoothness_constant
        self.proximal_operator = proximal_operator


class LinearMap:
    def __init__(self, lin_map, lin_map_adj, lin_map_norm=None):
        self.lin_map = lin_map
        self.lin_map_adj = lin_map_adj
        self.lin_map_norm = lin_map_norm


class ProblemData(NamedTuple):
    f: ObjectiveFunction
    g: ObjectiveFunction = None
    L: LinearMap = None
    x_true: np.ndarray = None
    observed_signal: np.ndarray = None


class Parameters(NamedTuple):
    initial_point: np.ndarray
    initial_point_dual: np.ndarray = None
    step_size: float = None
    step_size_dual: float = None
    is_fast: bool = False
    cp_par: float = None
    max_iter: int = None
    obj_decrease_tol: float = None
    grad_norm_tol: float = None
    output: bool = False


class Results(NamedTuple):
    opt_sol: np.ndarray
    obj_val: np.ndarray


# Implementation of the Bregman proximal gradient method
def bpg(data: ProblemData, pars: Parameters) -> Results:
    # Initialization and input validation
    iter_index = 0
    x = np.copy(pars.initial_point)

    if pars.step_size is None and data.f.smoothness_constant is not None:
        step_size = 1 / data.f.smoothness_constant
    elif pars.step_size is None and data.f.smoothness_constant is None:
        RuntimeError(
            'The step size or the smoothness constant of the function f must be specified.')
    else:
        step_size = pars.step_size

    if data.g is not None:
        if data.g.proximal_operator is None:
            raise RuntimeError(
                'The proximal operator of the function g must be specified.')
        x = data.g.proximal_operator(x, step_size)

    if pars.max_iter is not None:
        obj_val = np.zeros(pars.max_iter)
    else:
        obj_val = np.zeros(1)

    if data.f.residual is None:
        obj_val[iter_index] = data.f.val(x)
    else:
        r = data.f.residual(x)
        obj_val[iter_index] = data.f.val(r, True)

    if data.g is not None:
        obj_val[iter_index] += data.g.val(x)

    if pars.output:
        print("Bregman Proximal Gradient output:")
        print("+-------------+--------------")
        print("|  iter_index |      obj_val      |")
        print("+-------------+--------------")
        print("| %5d |     %5.9f |" % (iter_index, obj_val[iter_index]))

    # Main Loop
    stopping_criteria_flag = True
    while stopping_criteria_flag:
        iter_index += 1
        if data.f.residual is None:
            x = data.f.kernel.grad_dual(data.f.kernel.grad(x) - step_size * data.f.grad(x))
        else:
            x = data.f.kernel.grad_dual(data.f.kernel.grad(x) - step_size * data.f.grad(r, True))

        if data.g is not None:
            x = data.g.proximal_operator(x, step_size)

        if data.f.residual is not None:
            r = data.f.residual(x)

        if pars.max_iter is not None:
            iter_output = iter_index
            if iter_index >= pars.max_iter - 1:
                stopping_criteria_flag = False
        else:
            iter_output = 0

        if data.f.residual is None:
            obj_val[iter_output] = data.f.val(x)
        else:
            obj_val[iter_output] = data.f.val(r, True)

        if data.g is not None:
            obj_val[iter_output] += data.g.val(x)

        if pars.output:
            print("| %5d |     %5.9f   |" % (iter_index, obj_val[iter_output]))

    if pars.output:
        print("-----------------------------")

    return Results(x, obj_val)


# Implementation of the Chambolle-Pock method
def cp(data: ProblemData, pars: Parameters) -> Results:
    # Initialization and input validation
    iter_index = 0
    x = np.copy(pars.initial_point)
    x_bar = np.copy(pars.initial_point)
    x_prev = np.copy(pars.initial_point)

    if pars.cp_par is None:
        cp_par = 1
    elif pars.cp_par < 0:
        raise RuntimeError(
            'The Chambolle-Pock parameter (cp_par) must be non-negative.')
    else:
        cp_par = pars.cp_par

    if pars.initial_point_dual is None:
        raise RuntimeError('The dual initial point must be specified.')

    y = np.copy(pars.initial_point_dual)

    lin_map_norm = data.L.lin_map_norm
    if lin_map_norm is None and (pars.step_size is None or pars.step_size_dual is None):
        lin_map_norm = power_method(data.L.lin_map, data.L.lin_map_adj, x.shape)

    if pars.step_size is None and pars.step_size_dual is None:
        step_size = 0.49 / (lin_map_norm ** 2)
        step_size_dual = step_size
    elif pars.step_size is None:
        if pars.step_size_dual * (lin_map_norm ** 2) < 1:
            step_size_dual = pars.step_size_dual
            step_size = 0.99 / (step_size_dual * (lin_map_norm ** 2))
        else:
            raise RuntimeError(
                'It must hold that (step_size*step_size_dual*(L.lin_map_norm ** 2) < 1).')
    elif pars.step_size_dual is None:
        if pars.step_size * (lin_map_norm ** 2) < 1:
            step_size = pars.step_size
            step_size_dual = 0.99 / (step_size * (lin_map_norm ** 2))
        else:
            raise RuntimeError(
                'It must hold that (step_size*step_size_dual*(L.lin_map_norm ** 2) < 1).')
    else:
        step_size = pars.step_size
        step_size_dual = pars.step_size_dual

    if data.f.proximal_operator is None:
        raise RuntimeError(
            'The proximal operator of the function f must be specified.')

    if data.g is None or data.L is None:
        raise RuntimeError(
            'The function g and the linear mapping L must be specified.')

    if data.g.proximal_operator is None:
        raise RuntimeError(
            'The proximal operator of the function g must be specified.')

    # Compute the proximal operator of the conjugate of g using the extended Moreau decomposition
    def g_dual_proximal_operator(_y, _step_size_dual):
        return _y - _step_size_dual * data.g.proximal_operator(_y / _step_size_dual, 1 / _step_size_dual)

    if pars.max_iter is not None:
        obj_val = np.zeros(pars.max_iter)
    else:
        obj_val = np.zeros(1)

    obj_val[iter_index] = data.f.val(x) + data.g.val(data.L.lin_map(x))
    # Remark: The objective value is expected to be inf unless the functions f and g are finite.

    if pars.output:
        print("Chambolle-Pock method output:")
        print("+-------------+--------------")
        print("|  iter_index |      obj_val      |")
        print("+-------------+--------------")
        print("| %5d |     %5.9f |" % (iter_index, obj_val[iter_index]))

    # Main Loop
    stopping_criteria_flag = True
    while stopping_criteria_flag:
        iter_index += 1
        x_prev = np.copy(x)

        y = g_dual_proximal_operator(y + step_size_dual * data.L.lin_map(x_bar), step_size_dual)
        x = data.f.proximal_operator(x - step_size * data.L.lin_map_adj(y), step_size)
        x_bar = x + cp_par * (x - x_prev)

        if pars.max_iter is not None:
            iter_output = iter_index
            if iter_index >= pars.max_iter - 1:
                stopping_criteria_flag = False
        else:
            iter_output = 0

        obj_val[iter_output] = data.f.val(x) + data.g.val(data.L.lin_map(x))

        if pars.output:
            print("| %5d |     %5.9f   |" % (iter_index, obj_val[iter_output]))

    if pars.max_iter is None:
        obj_val[0] = data.f.val(x) + data.g.val(data.L.lin_map(x))

    if pars.output:
        print("-----------------------------")

    return Results(x, obj_val)
