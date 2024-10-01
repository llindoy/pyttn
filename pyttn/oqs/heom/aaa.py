import numpy as np
from scipy import sparse
from scipy import linalg as splinalg
from scipy.integrate import quad, quad_vec





def residue_integrand(theta, r, rs, poles):
    zvs = rs*np.exp(1.0j*theta)
    vals = r(poles+zvs)*zvs
    return vals

def compute_residues_integ(r,  poles, tol):
    #find the distance between pole i and the nearest pole to it - this ensures that we can evaluate the pole using
    vals = np.abs([xs - poles[np.argpartition(np.abs(poles - xs), 1)[1]] for xs in poles])/10.0
    rs = vals

    res = quad_vec(lambda x : residue_integrand(x, r, rs, poles), 0, 2.0*np.pi, epsrel = tol)[0]
    return res

#def compute_residues_rational(zeros, poles):

def prz(r, z, f, w, tol):

    m = w.shape[0]
    
    #setup the generalised eigenproblem suitable for obtaining the poles of this function
    B = np.identity(m+1, dtype = np.complex128)
    B[0, 0] = 0
    M = np.zeros((m+1, m+1), dtype = np.complex128)
    M[1:, 1:] = np.diag(z)
    M[0, 1:] = w
    M[1:, 0] = np.ones(m, dtype=np.complex128)

    poles = splinalg.eigvals(M, B)
    poles = poles[~np.isinf(poles)]

    #setup the generalised eigenproblem suitable for obtaining the zeros of this function
    M[0, 1:] = w*f

    zeros = splinalg.eigvals(M, B)
    zeros = zeros[~np.isinf(zeros)]

    res = compute_residues_integ(r, poles, tol)
    
    return poles, res, zeros

#function for evaluating the baryocentric form of the rational function approximation of another function
def evaluate_function(z, Z, f, w):
    ZZ, zz = np.meshgrid(Z, z)
    CC = 1.0/(zz-ZZ)
    r = (CC@(w*f))/(CC@w)
    return r

def AAA_algorithm(F, Z, tol=1e-13, nmax = 100, *args):
    M = Z.shape[0]

    #evaluate the function at the sample points
    Fz = np.array(F(Z, *args), dtype=np.complex128)
    Z = np.array(Z, dtype=np.complex128)
    Z0 = Z
    F0 = Fz

    R = np.mean(Fz)
    SF = sparse.diags(Fz)
    z = []
    f = []
    C = np.zeros( (M, nmax), dtype=np.complex128)
    w = None
    for i in range(nmax):
        ind = np.argmax(np.abs(Fz-R))
        z.append(Z[ind])
        f.append(Fz[ind])

        #delete the elements that we don't need any more
        Z = np.delete(Z, ind, 0)
        Fz = np.delete(Fz, ind, 0)
        C = np.delete(C, ind, 0)

        SF = sparse.diags(Fz)

        C[:, i] = (1.0/(Z-z[i]))
        Sf = np.diag(f)
        A = SF @ C[:, :i+1] - C[:, :i+1] @ Sf

        U, S, V = np.linalg.svd(A, full_matrices=False)
        V = np.conjugate(np.transpose(V))
        w = V[:, i]
        N = C[:, :i+1] @ (w*f)
        D = C[:, :i+1] @ w
        R = N/D
        err = np.linalg.norm(Fz-R, np.inf)
        if( err <= tol*np.linalg.norm(Fz, np.inf)):
            break

    z1 = np.array(z, dtype = np.complex128)
    f1 = np.array(f, dtype = np.complex128)
    w1 = np.array(w, dtype = np.complex128) 

    func = lambda x : evaluate_function(x, z1, f1, w1)
    poles,residues, zeros = prz(func, z, f, w, tol)
    
    return func, poles, residues, zeros


def C(dk, zk, t):
    Z, T = np.meshgrid(zk, t)
    return np.exp(-Z*T)@dk


