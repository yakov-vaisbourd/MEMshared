'''
Author:  Yakov Vaisbourd 2021-2022 with contributions from Ariel Goodwin 2021
'''


import memmpy.misc as misc

np = misc.np
find_root = misc.find_root

# super classes

class generic_dist(object):
    def __init__(self):
        pass

    def _validateArguments(self, arg, arg_name, validation_type=[], arg2=None, arg2_name=None):
        if not np.isscalar(arg) and not isinstance(arg, np.ndarray):
            raise ValueError('Argument ' + arg_name + ' must be a scalar or an ndarray (numpy).')

        if not np.all(np.isfinite(arg)):
            raise ValueError('Argument ' + arg_name + ' must contain only finite numbers.')

        validation_type = [validation_type] if isinstance(validation_type, str) else validation_type
        for val_type in validation_type:
            match val_type:
                case 'scalar':
                    if not np.isscalar(arg) and not np.size(arg) == 1:
                        raise ValueError('Argument ' + arg_name + ' must be a scalar.')
                case 'positive':
                    if np.any(arg <= 0):
                        raise ValueError('Argument ' + arg_name + ' must be positive in all entries.')
                case 'non-negative':
                    if np.any(arg < 0):
                        raise ValueError('Argument ' + arg_name + ' must be non-negative in all entries.')
                case 'positive scalar':
                    if (not np.isscalar(arg) and not np.size(arg) == 1) or np.any(arg <= 0):
                        raise ValueError('Argument ' + arg_name + ' must be a positive scalar.')
                case 'simplex':
                    if arg2 is None:
                        arg2 = 1
                    if np.any(arg < 0) or (np.sum(arg) != arg2):
                        raise ValueError(
                            'Argument ' + arg_name + ' must be in the ' + arg2 +
                            ' simplex (non-negative values which sum to ' + arg2 + ').')
                case 'simplex_ri':
                    if arg2 is None:
                        arg2 = 1
                    if np.any(arg <= 0) or (np.sum(arg) != arg2):
                        raise ValueError(
                            'Argument ' + arg_name + ' must be in the ' + arg2 +
                            ' simplex relative interior (positive values which sum to ' + arg2 + ').')
                case 'integer':
                    if np.any([not isinstance(par_i, int) and
                               not (isinstance(par_i, float) and par_i.is_integer()) for par_i in arg]):
                        raise ValueError('Argument ' + arg_name + ' must be an integer in all entries.')
                case 'dimensions':
                    if not (np.isscalar(arg) and np.isscalar(arg2)) and (
                            (not np.isscalar(arg) and not np.isscalar(arg2)) and arg.shape != arg2.shape):
                        raise ValueError('Arguments ' + arg_name + ' and ' + arg2_name +
                             ' must be scalars or one dimensional ndarray of the same size.')
                case 'ordered':
                    if np.any(arg2 < arg):
                        raise ValueError('All components of argument ' + arg_name + ' must be strictly smaller than  '
                                                                        '' + arg2_name + '.')
                case _:
                    return NotImplementedError

    def _verifyPars(self, pars, is_map=False):
        isNone = [par is None for par in pars]
        if any(isNone) and not all(isNone):
            raise ValueError('All or none of the arguments must be not None.')

        if not is_map:
            if not any(isNone):
                return [np.atleast_1d(par) for par in pars]
            else:
                return pars

    def cramer(self, x):
        raise NotImplementedError

    # bregman proximal operator generators - returns a function breg_prox(x,t)
    def bregman_prox_gen(self, x, t):
        raise NotImplementedError

    def kernel_val(self, x):
        raise NotImplementedError

    def kernel_grad(self, x):
        raise NotImplementedError

    def kernel_grad_dual(self, z):
        raise NotImplementedError


# Distributions

class normal_gen(generic_dist):
    def __init__(self, mu=None, sigma=None):
        mu, sigma = super()._verifyPars((mu, sigma))

        if mu is not None:
            self._validateArguments(mu, sigma)

        self.mu = mu
        self.sigma = sigma
        super().__init__()

    def _validateArguments(self, mu, sigma):
        super()._validateArguments(mu, 'mu')
        super()._validateArguments(sigma, 'sigma', 'positive')
        super()._validateArguments(mu, 'mu', 'dimensions', sigma, 'sigma')

    def _verifyPars(self, mu, sigma):
        mu, sigma = super()._verifyPars((mu, sigma))

        if mu is None:
            if self.mu is not None:
                mu = self.mu
                sigma = self.sigma
            else:
                raise ValueError('Distribution parameters were not specified.')

        self._validateArguments(mu, sigma)

        return mu, sigma

    def freeze(self, mu=None, sigma=None):
        mu, sigma = self._verifyPars(mu, sigma)
        self.mu = mu
        self.sigma = sigma
        return self

    def cramer(self, x, mu=None, sigma=None):
        mu, sigma = self._verifyPars(mu, sigma)
        super()._validateArguments(x, 'x', 'dimensions', mu, 'mu')
        return np.sum(((x - mu) ** 2) / (2 * sigma))

    # bregman proximal operator generators - returns a function breg_prox(x,t)
    def bregman_prox_gen(self, kernel, mu=None, sigma=None):
        mu, sigma = self._verifyPars(mu, sigma)
        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    return (sigma * x + t * mu) / (t + sigma)
            case 'Poisson':
                def breg_prox(x, t):
                    return 999
            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    return (mu * t / sigma - 1 / x + np.sqrt((mu * t / sigma - 1 / x) ** 2 + 4 * t / sigma)) / (
                            2 * t / sigma)
            case _:
                raise NotImplementedError

        return breg_prox

    def kernel_val(self, x):
        return x.dot(x) / 2

    def kernel_grad(self, x):
        return x

    def kernel_grad_dual(self, z):
        return z

normal = normal_gen()

class multnormal_gen(generic_dist):
    # Assumption: Denote by \Sigma a positive definite covariance matrix (default: \Sigma = I). The following
    # operators are used within the multnoral_gen class:
    #  - cov_mat(x) - yielding the product \Sigma x
    #  - cov_mat_inv(x) - yielding the product \Sigma^{-1}x
    #  - res_cov_mat(x, rho) - resolvent of the covariance matrix, yielding (rho I+\Sigma)^{-1} x

    def __init__(self, mu=None, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        mu, = super()._verifyPars((mu, ))
        super()._verifyPars((cov_mat, cov_mat_inv, res_cov_mat), is_map=True)
        if (mu is None and cov_mat is not None) or (mu is not None and cov_mat is None):
            raise ValueError('All or none of the arguments must be not None.')

        is_cov_mat_identity = False
        sigma = None
        if mu is not None:
            if cov_mat is None:
                cov_mat = lambda x: x
                cov_mat_inv = lambda x: x
                res_cov_mat = lambda x, rho: x * rho / (1 + rho)
                is_cov_mat_identity = True
                sigma = 1
            else:
                v = np.random.rand(*mu.shape)
                if np.all(cov_mat(v) / np.linalg.norm(cov_mat(v)) == v / np.linalg.norm(v)):
                    is_cov_mat_identity = True
                    sigma = cov_mat(v)[0] / v[0]
                    if sigma <= 0:
                        raise ValueError('The covariance matrix must be positive definite')

            self._validateArguments(mu, cov_mat, cov_mat_inv, res_cov_mat)

        self.mu = mu
        self.cov_mat_inv = cov_mat_inv
        self.res_cov_mat = res_cov_mat
        self.is_cov_mat_identity = is_cov_mat_identity
        self.sigma = sigma

        super().__init__()

    def _validateArguments(self, mu, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        super()._validateArguments(mu, 'mu')
        super()._verifyPars((cov_mat, cov_mat_inv, res_cov_mat), is_map=True)

    def _verifyPars(self, mu, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        mu, = super()._verifyPars((mu, ))
        if (mu is None and cov_mat is not None) or (mu is not None and cov_mat is None):
            raise ValueError('All or none of the arguments must be not None.')

        is_cov_mat_identity = False
        sigma = None
        if mu is not None:
            if cov_mat is None:
                cov_mat = lambda x: x
                cov_mat_inv = lambda x: x
                res_cov_mat = lambda x, rho: x * rho / (1 + rho)
                is_cov_mat_identity = True
                sigma = 1
            else:
                v = np.random.rand(*mu.shape)
                if np.all(cov_mat(v) / np.linalg.norm(cov_mat(v)) == v / np.linalg.norm(v)):
                    is_cov_mat_identity = True
                    sigma = cov_mat(mu)[0] / mu[0]
                    if sigma <= 0:
                        raise ValueError('The covariance matrix must be positive definite')

        else:
            if self.mu is not None:
                mu = self.mu
                cov_mat = self.cov_mat
                cov_mat_inv = self.cov_mat_inv
                res_cov_mat = self.res_cov_mat
                is_cov_mat_identity = self.is_cov_mat_identity
                sigma = self.sigma
            else:
                raise ValueError('Distribution parameters were not specified.')

        self._validateArguments(mu, cov_mat, cov_mat_inv, res_cov_mat)

        return mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma

    def freeze(self, mu=None, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma = \
            self._verifyPars(mu, cov_mat, cov_mat_inv, res_cov_mat)

        self.mu = mu
        self.cov_mat = cov_mat
        self.cov_mat_inv = cov_mat_inv
        self.res_cov_mat = res_cov_mat
        self.is_cov_mat_identity = is_cov_mat_identity
        self.sigma = sigma
        return self

    def cramer(self, x, mu=None, cov_mat = None, cov_mat_inv = None, res_cov_mat = None):
        mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma = \
            self._verifyPars(mu, cov_mat, cov_mat_inv, res_cov_mat)
        return np.tensordot(x - mu, cov_mat_inv(x - mu), x.ndim) / 2


    # bregman proximal operator generators - returns a function breg_prox(x,t)
    def bregman_prox_gen(self, kernel, mu=None, cov_mat = None, cov_mat_inv = None, res_cov_mat = None):
        mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma = \
            self._verifyPars(mu, cov_mat, cov_mat_inv, res_cov_mat)
        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    return res_cov_mat(cov_mat(x) + t * mu, t)
            case 'Poisson':
                def breg_prox(x, t):
                    return 999
            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    raise NotImplementedError
            case _:
                raise NotImplementedError

        return breg_prox

multnormal = multnormal_gen()


class bernoulli_gen(generic_dist):
    def __init__(self, p=None):
        p, = super()._verifyPars((p,))

        if p is not None:
            self._validateArguments(p)

        self.p = p
        super().__init__()

    def _validateArguments(self, p):
        super()._validateArguments(p, 'p', 'non-negative')
        super()._validateArguments(1 - p, '1-p', 'non-negative')

    def _verifyPars(self, p):
        p, = super()._verifyPars((p,))
        if p is None:
            if self.p is not None:
                p = self.p
            else:
                raise ValueError('Distribution parameters were not specified.')

        self._validateArguments(p)
        return p

    def freeze(self, p=None):
        p = self._verifyPars(p)
        self.p = p
        return self

    def cramer(self, x, p=None):
        p = self._verifyPars(p)
        p_is_one = p == 1
        p_is_zero = p == 0
        x = np.atleast_1d(x)

        if np.any(x < 0) or np.any(x > 1) or np.any(x[p_is_one] < 1) or np.any(x[p_is_zero] > 0):
            return np.inf

        x_is_one = x == 1
        x_is_zero = x == 0
        x_is_not_one_or_zero = np.logical_not(np.logical_or(x_is_one, x_is_zero))

        return (np.sum(x[x_is_not_one_or_zero] * np.log(x[x_is_not_one_or_zero] / p[x_is_not_one_or_zero])
                       + (1 - x[x_is_not_one_or_zero])
                       * np.log((1 - x[x_is_not_one_or_zero]) / (1 - p[x_is_not_one_or_zero])))
                - np.sum(np.log((1 - p[np.logical_and(x_is_zero, np.logical_not(np.logical_or(p_is_zero, p_is_one)))])))
                - np.sum(np.log(p[np.logical_and(x_is_one, np.logical_not(np.logical_or(p_is_zero, p_is_one)))])))

    def bregman_prox_gen(self, kernel, p=None):
        p = self._verifyPars(p)
        # p = np.atleast_1d(p)
        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    res = np.zeros_like(x)
                    res[p == 1] = 1
                    p_in = np.logical_and(0 < p, p < 1)
                    p_in_sum = p_in.sum()

                    def func_val_der(u, index_not_converged):
                        val = np.empty(p_in_sum, dtype=float)
                        np.copyto(val, (u - x[p_in]) / t - np.log(p[p_in] * (1 - u) / (u * (1 - p[p_in]))),
                                  where=index_not_converged)
                        der = np.empty(p_in_sum, dtype=float)
                        np.copyto(der, 1 / t + 1 / (u * (1 - u)), where=index_not_converged)
                        return val[index_not_converged], der[index_not_converged]

                    initial_point = np.full(p_in_sum, 0.5)
                    initial_point[x[p_in] == p[p_in]] = p[p_in][x[p_in] == p[p_in]]

                    res[p_in] = find_root(func_val_der, lower_bound=np.zeros(p_in_sum), upper_bound=np.ones(p_in_sum),
                                          initial_point=initial_point, max_iter=100, tol=1e-12)

                    return res

            case 'Poisson':
                def breg_prox(x, t):
                    return 999

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    res = np.ones_like(x)
                    p_in = np.logical_and(0 < p, p < 1)
                    p_in_sum = p_in.sum()

                    def func_val_der(u, index_not_converged):
                        val = np.empty(p_in_sum, dtype=float)
                        np.copyto(val, 1 / (t * x[p_in]) - 1 / (t * u)
                                  - np.log(p[p_in] * (1 - u) / (u * (1 - p[p_in]))), where=index_not_converged)
                        der = np.empty(p_in_sum, dtype=float)
                        np.copyto(der, 1 / (t * (u ** 2)) + 1 / (u * (1 - u)), where=index_not_converged)
                        return val[index_not_converged], der[index_not_converged]

                    initial_point = np.full(p_in_sum, 0.5)
                    initial_point[x[p_in] == p[p_in]] = p[p_in][x[p_in] == p[p_in]]

                    res[p_in] = find_root(func_val_der, lower_bound=np.zeros(p_in_sum),
                                          upper_bound=np.ones(p_in_sum), initial_point=initial_point,
                                          max_iter=100, tol=1e-12)

                    return res

            case _:
                raise NotImplementedError

        return breg_prox

    def kernel_val(self, x):
        raise NotImplementedError

    def kernel_grad(self, x):
        raise NotImplementedError

    def kernel_grad_dual(self, z):
        raise NotImplementedError

bernoulli = bernoulli_gen()


class norminvgauss_gen(generic_dist):
    def __init__(self, mu=None, alpha=None, beta=None, delta=None):
        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))

        if mu is not None:
            self._validateArguments(mu, alpha, beta, delta)

        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        super().__init__()

    def _validateArguments(self, mu, alpha, beta, delta):
        super()._validateArguments(mu, 'mu')
        super()._validateArguments(alpha, 'alpha')
        super()._validateArguments(beta, 'beta')
        super()._validateArguments(delta, 'delta', 'positive')
        super()._validateArguments(mu, 'mu', 'dimensions', alpha, 'alpha')
        super()._validateArguments(mu, 'mu', 'dimensions', beta, 'beta')
        super()._validateArguments(mu, 'mu', 'dimensions', delta, 'delta')
        if np.any(alpha < np.abs(beta)):
            raise ValueError('The arguments must satisfy alpha>=|beta|.')

    def _verifyPars(self, mu, alpha, beta, delta):
        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))

        if mu is None:
            if self.mu is not None:
                mu = self.mu
                alpha = self.alpha
                beta = self.beta
                delta = self.delta
            else:
                raise ValueError('Distribution parameters were not specified.')

        self._validateArguments(mu, alpha, beta, delta)

        return mu, alpha, beta, delta

    def freeze(self, mu=None, alpha=None, beta=None, delta=None):
        mu, alpha, beta, delta = self._verifyPars(mu, alpha, beta, delta)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        return self

    def cramer(self, x, mu=None, alpha=None, beta=None, delta=None):
        mu, alpha, beta, delta = self._verifyPars(mu, alpha, beta, delta)
        return np.sum(
            alpha * np.sqrt(delta ** 2 + (x - mu) ** 2) - beta * (x - mu) - delta * np.sqrt(alpha ** 2 - beta ** 2))

    def bregman_prox_gen(self, kernel, mu=None, alpha=None, beta=None, delta=None):
        mu, alpha, beta, delta = self._verifyPars(mu, alpha, beta, delta)
        mean_val = mu + delta * beta / np.sqrt(alpha ** 2 - beta ** 2)
        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    aux1 = (t * beta + x - mu) ** 2

                    def func_val_der(u, index_not_converged):
                        val = np.empty_like(mu, dtype=float)
                        np.copyto(val, (delta * u) ** 2 + aux1 * ((u / (1 + u)) ** 2) - (alpha * t) ** 2,
                                  where=index_not_converged)

                        der = np.empty_like(mu, dtype=float)
                        np.copyto(der, 2 * u * (delta ** 2) + 2 * aux1 * (u / ((1 + u) ** 3)),
                                  where=index_not_converged)

                        return val[index_not_converged], der[index_not_converged]

                    lb_val = np.zeros_like(x)
                    lb_index = aux1 <= (alpha * t) ** 2
                    lb_val[lb_index] = np.sqrt((alpha[lb_index] * t) ** 2 - aux1[lb_index]) / delta[lb_index]
                    initial_point = (lb_val + alpha * t / delta) / 2
                    x_eq_mean_val = x == mean_val
                    initial_point[x_eq_mean_val] = t * (np.sqrt(alpha[x_eq_mean_val] ** 2 - beta[x_eq_mean_val] ** 2)
                                                        / delta[x_eq_mean_val])

                    rho = find_root(func_val_der, lower_bound=lb_val, upper_bound=alpha * t / delta,
                                    initial_point=initial_point, max_iter=100, tol=1e-12)

                    return (t * beta + x + rho * mu) / (1 + rho)

            case 'Poisson':
                def breg_prox(x, t):
                    return 999
            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    aux1 = (t * beta - 1 / x)

                    def func_val_der(u, index_not_converged):
                        aux2 = np.sqrt(delta ** 2 + (u - mu) ** 2)

                        val = np.empty_like(mu, dtype=float)
                        np.copyto(val, t * alpha * (u - mu) * u - (aux1 * u + 1) * aux2,
                                  where=index_not_converged)

                        der = np.empty_like(mu, dtype=float)
                        np.copyto(der, t * alpha * (2 * u - mu) - aux1 * aux2 - (aux1 * u + 1) * (u - mu) / aux2,
                                  where=index_not_converged)

                        return val[index_not_converged], der[index_not_converged]

                    initial_point = (np.maximum(np.minimum(mean_val, x), 0)
                                     + np.maximum(np.maximum(mean_val, x), 1)) / 2
                    x_eq_mean_val = x == mean_val
                    initial_point[x_eq_mean_val] = x[x_eq_mean_val]

                    return find_root(func_val_der, lower_bound=np.maximum(np.minimum(mean_val, x), 0),
                                     upper_bound=np.maximum(np.maximum(mean_val, x), 1), initial_point=initial_point,
                                     max_iter=100, tol=1e-12)

            case _:
                raise NotImplementedError

        return breg_prox

norminvgauss = norminvgauss_gen()

class multnorminvgauss_gen(generic_dist):
    # Assumption: Denote by \Sigma a positive definite covariance matrix (default: \Sigma = I). The following
    # operators are used within the multnorminvgauss_gen class:
    #  - cov_mat(x) - yielding the product \Sigma x
    #  - cov_mat_inv(x) - yielding the product \Sigma^{-1}x
    #  - res_cov_mat_inv(x, \rho) - resolvent of the covariance inverse, yielding (\rho^{-1}I+\Sigma^{-1})^{-1} x

    def __init__(self, mu=None, alpha=None, beta=None, delta=None, cov_mat=None, cov_mat_inv=None,
                 res_cov_mat_inv=None):
        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))
        super()._verifyPars((cov_mat, cov_mat_inv, res_cov_mat_inv), is_map=True)
        if mu is None and cov_mat is not None:
            raise ValueError('Covariance mappings can be defined only if the standard parameters were specified.')

        is_cov_mat_identity = False
        sigma = None
        if mu is not None:
            if cov_mat is None:
                cov_mat = lambda x: x
                cov_mat_inv = lambda x: x
                res_cov_mat_inv = lambda x, rho: x * rho / (1 + rho)
                is_cov_mat_identity = True
                sigma = 1
            else:
                if np.all(cov_mat(mu) / np.linalg.norm(cov_mat(mu)) == mu / np.linalg.norm(mu)):
                    is_cov_mat_identity = True
                    sigma = cov_mat(mu)[0] / mu[0]
                    if sigma <= 0:
                        raise ValueError('The covariance matrix must be positive definite')

            self._validateArguments(mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv)

        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.cov_mat = cov_mat
        self.cov_mat_inv = cov_mat_inv
        self.res_cov_mat_inv = res_cov_mat_inv
        self.is_cov_mat_identity = is_cov_mat_identity
        self.sigma = sigma

        if mu is not None:
            self._gamma = np.sqrt(alpha ** 2 - np.tensordot(beta, cov_mat(beta), beta.ndim))
        else:
            self._gamma = None

        super().__init__()

    def _validateArguments(self, mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv):
        super()._validateArguments(mu, 'mu')
        super()._validateArguments(alpha, 'alpha', 'scalar')
        super()._validateArguments(beta, 'beta')
        super()._validateArguments(delta, 'delta', 'positive scalar')
        super()._validateArguments(mu, 'mu', 'dimensions', beta, 'beta')

        if alpha ** 2 < np.tensordot(beta, cov_mat(beta), beta.ndim):
            raise ValueError('The arguments must satisfy alpha>=np.inner(beta, cov_mat(beta)).')

    def _verifyPars(self, mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv):
        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))
        super()._verifyPars((cov_mat, cov_mat_inv, res_cov_mat_inv), is_map=True)
        if mu is None and cov_mat is not None:
            raise ValueError('Covariance mappings can be defined only if the standard parameters were specified.')

        is_cov_mat_identity = False
        sigma = None
        if mu is not None:
            if cov_mat is None:
                cov_mat = lambda x: x
                cov_mat_inv = lambda x: x
                res_cov_mat_inv = lambda x, rho: x * rho / (1 + rho)
                is_cov_mat_identity = True
                sigma = 1
            else:
                if np.linalg.norm(cov_mat(mu) / np.linalg.norm(cov_mat(mu)) - mu / np.linalg.norm(mu)) < misc.TOLERANCE:
                    is_cov_mat_identity = True
                    sigma = cov_mat(mu)[0] / mu[0]
                    if sigma <= 0:
                        raise ValueError('The covariance matrix must be positive definite')

            _gamma = np.sqrt(alpha ** 2 - np.tensordot(beta, cov_mat(beta), beta.ndim))
        else:
        # if mu is None:
            if self.mu is not None:
                mu = self.mu
                alpha = self.alpha
                beta = self.beta
                delta = self.delta
                cov_mat = self.cov_mat
                cov_mat_inv = self.cov_mat_inv
                res_cov_mat_inv = self.res_cov_mat_inv
                _gamma = self._gamma
                is_cov_mat_identity = self.is_cov_mat_identity
                sigma = self.sigma
            else:
                raise ValueError('Distribution parameters were not specified.')

        self._validateArguments(mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv)

        return mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv, _gamma, is_cov_mat_identity, sigma

    def freeze(self, mu=None, alpha=None, beta=None, delta=None, cov_mat=None, cov_mat_inv=None,
               res_cov_mat_inv=None):
        mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv, _gamma, is_cov_mat_identity, sigma = \
            self._verifyPars(mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.cov_mat = cov_mat
        self.cov_mat_inv = cov_mat_inv
        self.res_cov_mat_inv = res_cov_mat_inv
        self._gamma = np.sqrt(alpha ** 2 - np.tensordot(beta, cov_mat(beta), beta.ndim))
        self.is_cov_mat_identity = is_cov_mat_identity
        self.sigma = sigma
        return self

    def cramer(self, x, mu=None, alpha=None, beta=None, delta=None,
               cov_mat=None, cov_mat_inv=None, res_cov_mat_inv=None):
        mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv, _gamma, is_cov_mat_identity, sigma = \
            self._verifyPars(mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv)
        x = np.atleast_1d(x)
        return (alpha * np.sqrt(delta ** 2 + np.tensordot(x - mu, cov_mat_inv(x - mu), x.ndim))
                - np.tensordot(beta, x - mu, x.ndim) - delta * _gamma)

    def bregman_prox_gen(self, kernel, mu=None, alpha=None, beta=None, delta=None,
                         cov_mat=None, cov_mat_inv=None, res_cov_mat_inv=None):
        mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv, _gamma, is_cov_mat_identity, sigma = \
            self._verifyPars(mu, alpha, beta, delta, cov_mat, cov_mat_inv, res_cov_mat_inv)

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    aux1 = t * beta + x - mu

                    if is_cov_mat_identity:
                        aux1 = np.linalg.norm(t * beta + x - mu) ** 2
                    else:
                        aux1 = t * beta + x - mu

                    def func_val_der(u, index_not_converged):
                        val = np.empty(1, dtype=float)
                        if is_cov_mat_identity:
                            np.copyto(val, (delta * u) ** 2 + aux1 * (sigma * u / (sigma + u)) ** 2 - (alpha * t) ** 2,
                                      where=index_not_converged)

                            der = np.empty(1, dtype=float)
                            np.copyto(der, 2 * u * (delta ** 2) + 2 * aux1 * u * ( sigma / (sigma + u)) ** 3,
                                      where=index_not_converged)
                            der_out = der[index_not_converged]
                        else:
                            aux2 = res_cov_mat_inv(aux1, u)
                            np.copyto(val, (delta * u) ** 2 + np.tensordot(aux2, cov_mat_inv(aux2), aux2.ndim)
                                      - (alpha * t) ** 2, where=index_not_converged)

                            der_out = None

                        return val[index_not_converged], der_out

                    rho = find_root(func_val_der, lower_bound=np.zeros(1), upper_bound=alpha * t / delta,
                                    initial_point=alpha * t / (2 * delta), max_iter=100, tol=1e-12)

                    return res_cov_mat_inv(t * beta + x + rho * cov_mat_inv(mu), rho) / rho

            case 'Poisson':
                def breg_prox(x, t):
                    return 999

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    if not is_cov_mat_identity:
                        raise NotImplementedError

                    x = np.atleast_1d(x)
                    aux1 = (t * beta - 1 / x)

                    def func_val_der(u, index_not_converged):
                        aux2 = np.sqrt((aux1 + mu * u) ** 2 + 4 * u)

                        val = np.empty(1, dtype=float)
                        np.copyto(val, (delta * u) ** 2 + np.sum((aux1 + aux2 - mu * u) ** 2) / (4 * sigma)
                                  - (alpha * t / sigma) ** 2, where=index_not_converged)

                        der = np.empty(1, dtype=float)
                        np.copyto(der, 2 * (delta ** 2) * u + np.sum((aux1 + aux2 - mu * u)
                                * ((mu * (aux1 + mu * u) + 2) / aux2 - mu)) / (2 * sigma), where=index_not_converged)

                        return val[index_not_converged], der[index_not_converged]

                    rho = find_root(func_val_der, lower_bound=np.zeros(1), upper_bound=t * alpha / (sigma * delta),
                                    initial_point=t * alpha / (2 * sigma * delta), max_iter=100, tol=1e-12)

                    return (aux1 + mu * rho + np.sqrt((aux1 + mu * rho) ** 2 + 4 * rho)) / (2 * rho)

            case _:
                raise NotImplementedError

        return breg_prox

multnorminvgauss = multnorminvgauss_gen()


class gamma_gen(generic_dist):
    def __init__(self, alpha=None, beta=None):
        alpha, beta = super()._verifyPars((alpha, beta))

        if alpha is not None:
            self._validateArguments(alpha, beta)

        self.alpha = alpha
        self.beta = beta
        super().__init__()

    def _validateArguments(self, alpha, beta):
        super()._validateArguments(alpha, 'alpha', 'positive')
        super()._validateArguments(beta, 'beta', 'positive')
        super()._validateArguments(alpha, 'alpha', 'dimensions', beta, 'beta')

    def _verifyPars(self, alpha, beta):
        alpha, beta = super()._verifyPars((alpha, beta))

        if alpha is None:
            if self.alpha is not None:
                alpha = self.alpha
                beta = self.beta
            else:
                raise ValueError('Distribution parameters were not specified.')

        self._validateArguments(alpha, beta)

        return alpha, beta

    def freeze(self, alpha=None, beta=None):
        alpha, beta = self._verifyPars(alpha, beta)
        self.alpha = alpha
        self.beta = beta
        return self

    def cramer(self, x, alpha=None, beta=None):
        alpha, beta = self._verifyPars(alpha, beta)
        return np.sum(beta * x - alpha + alpha * np.log(alpha / (beta * x)))

    def bregman_prox_gen(self, kernel, alpha=None, beta=None):
        alpha, beta = self._verifyPars(alpha, beta)

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    return (x - t * beta + np.sqrt((x - t * beta) ** 2 + 4 * t * alpha)) / 2

            case 'Poisson':
                def breg_prox(x, t):
                    return 999
            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    return x * (t * alpha + 1) / (x * t * beta + 1)

            case _:
                raise NotImplementedError

        return breg_prox

    def kernel_val(self, x):  # Assumption beta = 1
        return -np.sum(np.log(x))

    def kernel_grad(self, x):  # Assumption beta = 1
        return -1 / x

    def kernel_grad_dual(self, z):  # Assumption beta = 1
        return -1 / z

gamma = gamma_gen()


class poisson_gen(generic_dist):
    def __init__(self, _lambda=None):
        _lambda, = super()._verifyPars((_lambda,))

        if _lambda is not None:
            self._validateArguments(_lambda)

        self._lambda = _lambda
        super().__init__()

    def _validateArguments(self, _lambda):
        super()._validateArguments(_lambda, '_lambda', 'positive')

    def _verifyPars(self, _lambda):
        _lambda, = super()._verifyPars((_lambda,))

        if _lambda is None:
            if self._lambda is not None:
                _lambda = self._lambda
            else:
                raise ValueError('Distribution parameters were not specified.')

        self._validateArguments(_lambda)

        return _lambda

    def freeze(self, _lambda=None):
        _lambda = self._verifyPars(_lambda)
        self._lambda = _lambda
        return self

    def cramer(self, x, _lambda=None):
        _lambda = self._verifyPars(_lambda)
        return np.sum(x * np.log(x / _lambda) - x + _lambda)

    def bregman_prox_gen(self, kernel, _lambda=None):
        _lambda = self._verifyPars(_lambda)
        mean_val = _lambda

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        val = np.empty_like(_lambda, dtype=float)
                        np.copyto(val, np.log(u / _lambda) + (u - x) / t, where=index_not_converged)

                        der = np.empty_like(_lambda, dtype=float)
                        np.copyto(der, 1 / u + 1 / t, where=index_not_converged)

                        return val[index_not_converged], der[index_not_converged]

                    lb_val = np.maximum(np.minimum(mean_val, x), 0)
                    ub_val = np.maximum(np.maximum(mean_val, x), 1)
                    return find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                     initial_point=(lb_val + ub_val) / 2, max_iter=100, tol=1e-12)

            case 'Poisson':
                def breg_prox(x, t):
                    return 999
            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        val = np.empty_like(_lambda, dtype=float)
                        np.copyto(val, t * np.log(u / _lambda) - 1 / u + 1 / x, where=index_not_converged)

                        der = np.empty_like(_lambda, dtype=float)
                        np.copyto(der, t / u + 1 / (u ** 2), where=index_not_converged)

                        return val[index_not_converged], der[index_not_converged]

                    lb_val = np.maximum(np.minimum(mean_val, x), 0)
                    ub_val = np.maximum(np.maximum(mean_val, x), 1)
                    return find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                     initial_point=(lb_val + ub_val) / 2, max_iter=100, tol=1e-12)

            case _:
                raise NotImplementedError

        return breg_prox

    def kernel_val(self, x):  # Assumption beta = 1
        return np.sum(x * np.log(x))

    def kernel_grad(self, x):  # Assumption beta = 1
        return np.log(x) + 1

    def kernel_grad_dual(self, z):  # Assumption beta = 1
        return np.exp(z - 1)

poisson = poisson_gen()