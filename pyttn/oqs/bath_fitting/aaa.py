import numpy as np
from scipy import sparse
from scipy import linalg as splinalg
from scipy.integrate import quad_vec


def __residue_integrand(theta, r, rs, poles):
    zvs = rs * np.exp(1.0j * theta)
    vals = r(poles + zvs) * zvs
    return vals


def __compute_residues_integ(r, poles, tol):
    r"""Compute the poles and residues given the baryocentric representation of a rational function

    :param r: The rational function object
    :type r: callable
    :params poles: The poles of the rational funcction
    :type poles: np.ndarray
    :param tol: Integration tolerance for computing residues
    :type tol: float

    :returns: The residue associated with each pole
    :rtype: np.ndarray
    """
    # find the distance between pole i and the nearest pole to it - this ensures that we can evaluate the pole using
    vals = (
        np.abs([xs - poles[np.argpartition(np.abs(poles - xs), 1)[1]] for xs in poles])
        / 10.0
    )
    rs = vals

    res = quad_vec(
        lambda x: __residue_integrand(x, r, rs, poles), 0, 2.0 * np.pi, epsrel=tol
    )[0]
    return res


def __prz(r, z, f, w, tol):
    r"""Compute the poles and residues given the baryocentric representation of a rational function

    :param r: The rational function object
    :type r: callable
    :params z: Support points
    :type z: np.ndarray
    :params f: The data values
    :type f: np.ndarray
    :params w: The weights of the baryocentric approximation
    :type w: np.ndarray
    :param tol: Integration tolerance for computing residues
    :type tol: float

    :returns:
        - poles(np.ndarray) - The poles of the rational function
        - residues(np.ndarray) - The residues of the rational function decomposition
        - zeros(np.ndarray) - The zeros of the rational function
    """

    m = w.shape[0]

    # setup the generalised eigenproblem suitable for obtaining the poles of this function
    B = np.identity(m + 1, dtype=np.complex128)
    B[0, 0] = 0
    M = np.zeros((m + 1, m + 1), dtype=np.complex128)
    M[1:, 1:] = np.diag(z)
    M[0, 1:] = w
    M[1:, 0] = np.ones(m, dtype=np.complex128)

    poles = splinalg.eigvals(M, B)
    poles = poles[~np.isinf(poles)]

    # setup the generalised eigenproblem suitable for obtaining the zeros of this function
    M[0, 1:] = w * f

    zeros = splinalg.eigvals(M, B)
    zeros = zeros[~np.isinf(zeros)]

    res = __compute_residues_integ(r, poles, tol)

    return poles, res, zeros


# function for evaluating the baryocentric form of the rational function approximation of another function


def __evaluate_function(z, Z, f, w):
    r"""Evaluate the baryocentric form of the rational function approximation

    .. math:
        r(z) = \frac{\sum_{j=1}^N \frac{w_j f_j}{z-Z_j}}{\sum_{j=1}^N \frac{w_j}{z-Z_j}}

    :params z: The point at which to evaluate the function
    :type z: np.ndarray
    :params Z: Support points
    :type Z: np.ndarray
    :params f: The data values
    :type f: np.ndarray
    :params w: The weights of the baryocentric approximation
    :type w: np.ndarray

    :returns: The value of the approximation at the points z
    :rtype: np.ndarray
    """
    ZZ, zz = np.meshgrid(Z, z)
    CC = 1.0 / (zz - ZZ)
    r = (CC @ (w * f)) / (CC @ w)
    return r


def AAA_algorithm(F, Z, tol=1e-13, K=None, nmax=100, *args, **kwargs):
    r"""Implementation of the adaptive Antoulas-Anderson (AAA) algorithm for rational approximation
    Y. Nakatsukasa, O. Sète, and L. N. Trefethen, SIAM Journal on Scientific Computing 40, A1494 (2018).

    :param F: The function to be fit
    :type F: callable
    :param Z: The set of support points used for interpolating the function
    :type Z: np.ndarray
    :param tol: The convergence tolerance for the AAA algorithm. (default: 1e-13)
    :type tol: float, optional
    :param K: The maximum number of poles to fit. (default: None)
    :type K: int or None, optional
    :param nmax: The maximum number of poles to use in the AAA fit. (default: 100)
    :type nmax: int, optional
    :param *args: Variable length argument list to be passed to the F function
    :param **kwargs: Arbitrary keyword arguments to be passed to the F function

    :returns:
        - func - A function defining the rational function approximation
        - poles(np.ndarray) - The poles of the rational function
        - residues(np.ndarray) - The residues of the rational function decomposition
        - zeros(np.ndarray) - The zeros of the rational function

    """

    M = Z.shape[0]

    # evaluate the function at the sample points
    Fz = np.array(F(Z, *args, **kwargs), dtype=np.complex128)
    Z = np.array(Z, dtype=np.complex128)

    R = np.mean(Fz)
    SF = sparse.diags(Fz)
    z = []
    f = []
    C = np.zeros((M, nmax), dtype=np.complex128)
    w = None
    for i in range(nmax):
        ind = np.argmax(np.abs(Fz - R))
        z.append(Z[ind])
        f.append(Fz[ind])

        # delete the elements that we don't need any more
        Z = np.delete(Z, ind, 0)
        Fz = np.delete(Fz, ind, 0)
        C = np.delete(C, ind, 0)

        SF = sparse.diags(Fz)

        C[:, i] = 1.0 / (Z - z[i])
        Sf = np.diag(f)
        A = SF @ C[:, : i + 1] - C[:, : i + 1] @ Sf

        U, S, V = np.linalg.svd(A, full_matrices=False)
        V = np.conjugate(np.transpose(V))
        w = V[:, i]
        N = C[:, : i + 1] @ (w * f)
        D = C[:, : i + 1] @ w
        R = N / D
        err = np.linalg.norm(Fz - R, np.inf)
        if err <= tol * np.linalg.norm(Fz, np.inf) or len(z) == K:
            break

    z1 = np.array(z, dtype=np.complex128)
    f1 = np.array(f, dtype=np.complex128)
    w1 = np.array(w, dtype=np.complex128)

    def func(x):
        return __evaluate_function(x, z1, f1, w1)

    poles, residues, zeros = __prz(func, z, f, w, tol)

    return func, poles, residues, zeros
