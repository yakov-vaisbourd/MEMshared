"""
Author:  Yakov Vaisbourd  2021-2022

Image processing toolbox supplementing the memmpy package. The module includes basic operators for image deblurring.
The implementation is based on:
Deblurring Images: Matrices, Spectra, and Filtering by Per Christian Hansen, James G.Nagy, and Dianne P.O'Leary

"""

import numpy as np
from scipy import fftpack as fp


def psfGauss(dim, s=2):
    """
    Create a point spread function for Gaussian blur.

    :param dim: Dimension of the psf array.
                Two-dimensional vector indicating the row and column dimensions, respectively. Alternatively, a scalar
                indicating the row and column dimensions in case that they are the same.
    :param s: Standard deviation of the Gaussian.
                Two-dimensional vector indicating the standard deviation along the rows and columns, respectively.
                Alternatively, a scalar indicating the standard deviation along the rows and columns in case that
                they are the same (Default: 2).
    :return: psf - array containing the psf
             center - two-dimensional vector containing the index of the center of psf
    """

    dim = np.atleast_1d(np.asarray(dim))
    if dim.size == 1:
        m = dim
        n = dim
    else:
        m = dim[0]
        n = dim[1]

    s = np.atleast_1d(np.asarray(s))
    if s.size == 1:
        s = np.array([s, s])

    xx = np.arange(-(n // 2), - (-n // 2))
    yy = np.arange(-(m // 2), - (-m // 2))
    x, y = np.meshgrid(xx, yy)

    psf = np.exp(-(x ** 2) / (2 * s[0] ** 2) - (y ** 2) / (2 * s[1] ** 2))
    psf = psf / np.sum(psf)

    center = np.unravel_index(psf.argmax(), psf.shape)

    return psf, center


def test():
    return 1


def psfDefocus(dim, r=None):
    """
    Create a point spread function for out-of-focus blur.
    The point spread function for out-of-focus blur is defined as 1/(pi*r*r) inside a circle of radius r, and zero
    otherwise.


    :param dim: Dimension of the psf array.
                Two-dimensional vector indicating the row and column dimensions, respectively. Alternatively, a scalar
                indicating the row and column dimensions in case that they are the same.
    :param r: Radius of out-of-focus (Default: min(dim//2-1) ).
    :return: psf - array containing the psf
             center - two-dimensional vector containing the index of the center of psf
    """

    dim = np.atleast_1d(np.asarray(dim))
    if dim.size == 1:
        m = dim
        n = dim
    else:
        m = dim[0]
        n = dim[1]

    center = (np.array([m, n]) + 1) // 2 - 1
    if r is None:
        r = np.min(center)

    if r == 0:
        psf = np.zeros((m, n))
        psf[center] = 1
    else:
        psf = np.ones((m, n)) / (np.pi * r * r)

    x, y = np.meshgrid(np.arange(n), np.arange(m))
    idx = (x - center[1]) ** 2 + (y - center[0]) ** 2 > r ** 2
    psf[idx] = 0
    psf = psf / np.sum(psf)

    return psf, center


def padPSF(psf, m, n):
    """
    Pad psf with zeros to yield an m x n psf

    :param psf: Original point spread function
    :param m: Row dimension of the desired size
    :param n: Column dimension of the desired size
    :return: Padded psf of dimension m x n
    """

    PSF_big = np.zeros([m, n])
    PSF_big[0:psf.shape[0], 0:psf.shape[1]] = psf

    return PSF_big


def dctshift(psf, center):
    """
    Create the first column of the blurring matrix under reflexive boundary conditions.

    :param psf: Point spread function
    :param center: Center of the point spread function
    :return: A vector that contains the first column of the blurring matrix
    """
    m, n = psf.shape
    i, j = center
    k = np.min((i, m - 1 - i, j, n - 1 - j))

    truncated_psf = psf[(i - k):(i + k + 1), (j - k):(j + k + 1)]

    extract_mat_1 = np.diag(np.ones(k + 1), k)
    extract_mat_2 = np.diag(np.ones(k), k + 1)

    psf_res = np.zeros_like(psf)
    psf_res[0:(2 * k + 1), 0:(2 * k + 1)] = (np.dot(extract_mat_1, np.dot(truncated_psf, extract_mat_1.transpose()))
                                             + np.dot(extract_mat_1, np.dot(truncated_psf, extract_mat_2.transpose()))
                                             + np.dot(extract_mat_2, np.dot(truncated_psf, extract_mat_1.transpose()))
                                             + np.dot(extract_mat_2, np.dot(truncated_psf, extract_mat_2.transpose())))

    return psf_res


def spectral_decomposition_gen(psf, psf_center, img_shape, boundary_conditions='periodic'):
    """
    Create the components of the spectral decomposition of the blurring matrix A based on the point spread function,
    desired image size and boundary conditions. Namely, find matrix B and S such that: A = conj(B)*S*B

    :param psf: The point spread function
    :param img_shape: Desired image size
    :param psf_center: Center of the point spread function
    :param boundary_conditions: Boundary conditions (Default: periodic)
    :return spectra: The spectra matrix (S)
    :return basis_map: An operator that yield the product of the basis matrix B with a given vector (image)
    :return basis_map_adj: An operator that yield the product of the adjoint of the basis matrix B with a given vector
                            (image in the frequency domain)
    """

    img_shape = np.atleast_1d(np.asarray(img_shape))
    if img_shape.size == 1:
        img_shape = np.array([img_shape, img_shape])

    psf_big = padPSF(psf, *img_shape)

    if boundary_conditions == 'periodic':
        spectra = fp.fft2(np.roll(np.roll(psf_big, -psf_center[1], axis=1), -psf_center[0], axis=0))
        trans = lambda x: fp.fft2(x) / np.sqrt(x.size)
        itrans = lambda x: np.real(fp.ifft2(x) * np.sqrt(x.size))

    elif boundary_conditions == 'reflexive':
        e1 = np.zeros_like(psf_big)
        e1[0, 0] = 1
        spectra = fp.dctn(dctshift(psf_big, psf_center), norm='ortho') / fp.dctn(e1, norm='ortho')
        trans = lambda x: fp.dctn(x, norm='ortho')
        itrans = lambda x: fp.dctn(x, norm='ortho')
    else:
        raise ValueError('Supported boundary conditions (boundary_conditions) are: periodic, reflexive.')

    return spectra, trans, itrans


def dif_map(x, boundary_conditions='periodic'):
    if boundary_conditions == 'periodic':
        return np.vstack((np.vstack((x[0:-1, :] - x[1:, :], x[-1:, :] - x[:1, :])),
                          np.column_stack((x[:, 0:-1] - x[:, 1:], x[:, -1:] - x[:, :1]))))
    elif boundary_conditions == 'reflexive':
        return np.vstack((np.vstack((x[0:-1, :] - x[1:, :], np.zeros(x.shape[1]))),
                          np.column_stack((x[:, 0:-1] - x[:, 1:], np.zeros(x.shape[0])))))
    else:
        raise ValueError('Supported boundary conditions are: periodic, reflexive.')

    # zero bc:
    # return np.vstack((np.vstack((x[0:-1, :] - x[1:, :], x[-1:, :])), np.hstack((x[:, 0:-1] - x[:, 1:], x[:, -1:]))))


def dif_map_adj(y, boundary_conditions='periodic'):
    m = y.shape[0]//2
    if boundary_conditions == 'periodic':
        return (np.vstack((y[:1, :] - y[(m - 1):m, :], y[1:m, :] - y[0:(m - 1), :]))
                + np.column_stack((y[m:, :1] - y[m:, -1:], y[m:, 1:] - y[m:, 0:-1])))
    elif boundary_conditions == 'reflexive':
        return (np.vstack((y[0, :], y[1:(m - 1), :] - y[0:(m - 2), :], - y[(m-2):(m-1), :]))
                + np.column_stack((y[m:, 0], y[m:, 1:-1] - y[m:, 0:-2], - y[m:, -2:-1])))
    else:
        raise ValueError('Supported boundary conditions are: periodic, reflexive.')
