import numpy as np
import matplotlib.pyplot as plt
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
    #plt.plot(Z, Fz, 'k', linewidth=5)
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
        #if(err < 1e-2):
        #    plt.plot(Z, R)
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


def test():
    def Jw(w, rs, eps):#, wc, s):
        J = 0*w
        for i in range (rs.shape[0]):
            J = J + 1/( (w-i*0.1)*(w-i*0.1) + rs[i]*rs[i])#*np.where(w < eps, 1.0, 0.0)
        return J
        #return 1/(w*w+r*r) #np.where(np.abs(w) < np.abs(r), np.sqrt(r*r-w*w),  0.0)#np.pi/2.0*alpha*np.power(w, s)*np.power(wc, 1-s)*np.exp(-np.abs(w)/wc)


    def JwBO(w, O, g):
        return 15*2*w*g*O*O/((w*w-O*O)*(w*w-O*O) + g*g*O*O)
    
    def JwDebye(w, wc):
        return 2*w*wc/(w*w+wc*wc)
    
    def Jexp(w, alpha, wc):
        return np.where(w > 0.0, alpha*w*np.exp(-w/wc), 0.0)
    
    def Jw2(w, wc, tau, D):
        s = 0.5
        fw = None
        if(D == 1):
            fw = np.cos(w*tau)
        if(D == 3):
            fw = np.sin(w*tau)/(w*tau)
        return np.power(w, D)/np.power(wc, D-1)*np.exp(-np.abs(w/wc))*(1-fw)/2.0#+np.cos(w*2*tau))/4.0
        #return w*np.exp(-np.abs(w/wc))*(1+np.cos(w*tau))/2.0#+np.cos(w*2*tau))/4.0
        #return np.pi/2.0*tau*np.power(wc, 1-s)*np.power(np.abs(w), s)*np.exp(-np.abs(w)/wc)#*(1+np.cos(w*tau))/2.0
    
    def Sw(w, beta, J, *args):
        return J(w, *args)*0.5*(1+1.0/np.tanh(beta*w/2.0))
    
    def Sw0(w, J, *args):
        return np.where(w > 0, J(w, *args), 0)
    wc = 1
    g = wc*2
    
    #Z1 = 1-np.logspace(-8, -2, 100)
    #Z1 = np.concatenate((Z1, 1+np.logspace(-8, -2, 100)))
    Z1 = np.linspace(1e-8, 20, 2000)#np.concatenate((Z1, np.linspace(1e-8, 5, 2000)))
    #Z1 = 
    #Z1 = np.logspace(-2, np.log(1000)/np.log(10), 2000)#, np.linspace(1.01e-1, 20*wc, 1800)))
    #Z1 = np.concatenate((np.logspace(-8, -1, 200), np.linspace(1.01e-1, 20*wc, 1800)))
    #nZ1 = -np.concatenate((np.logspace(-8, -1, 200), np.linspace(1.01e-1, 20*wc, 100)))
    nZ1 = -np.flip(Z1)
    Z = np.concatenate((nZ1, Z1))
    
    alpha = 1.0
    wc = 1
    
    Zv = np.linspace(-20*wc, 20*wc, 100000)
    func1, p, r, z = AAA_algorithm(lambda x : Jexp(x, alpha, wc), Z, nmax = 500, tol=1e-4)
    print(p, r/(2.0*np.pi), z)
    p=1.0j*p
    Zv = np.linspace(-20*wc, 20*wc, 100000)
    plt.plot(Zv, func1(Zv)-Jexp(Zv, alpha, wc))
    plt.plot(Z, func1(Z)-Jexp(Z, alpha, wc))
    #plt.plot(Zv, )
    plt.show()
    
    fv = func1(Zv)
    f2v = Jexp(Zv, alpha, wc)
    plt.plot(Zv, np.real(f2v), 'k-')
    #plt.plot(Zv, np.imag(f2v), 'k--')
    plt.plot(Zv, np.real(fv), 'r--')
    #plt.plot(Zv, np.imag(fv), 'r--')
    plt.show()
    #func2 = AAA_algorithm(lambda x : Sw0(x,Jw2, 1, 10), Z, tol=1e-8,nmax=1000)
    #func3 = AAA_algorithm(lambda x : Sw0(x,Jw2, 1, 10), Z, tol=1e-12,nmax=1000)
    #plt.show()
    
    #Z = np.exp(np.linspace(-0.5,0.5+15.0j*np.pi,1000))
    #func = lambda x : np.tan(np.pi*x/2.0)
    #func1, p, r, z = AAA_algorithm(func, Z, tol=1e-13,nmax=100)
    #plt.show()
    #print(p)
    
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(Z.real, Z.imag, np.abs(func(Z)))
    #plt.show()
    #exit(1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = np.where(r<0, 'r', 'b')
    #ax.scatter(p.real, p.imag)#, np.log(np.abs(r)/(2.0*np.pi))/np.log(10), c=colors.ravel())
    ax.scatter(p.real, p.imag, np.log(np.abs(r))/np.log(10))
    
    #pd = np.genfromtxt("poles.csv")
    #poles2 = pd[:, 0] + pd[:, 1]*1.0j
    #res2 = pd[:, 2] + pd[:, 3]*1.0j
    #colors = np.where(res2<0, 'orange', 'cyan')
    #ax.scatter(poles2.real, poles2.imag)#, np.log(np.abs(res2))/np.log(10), c=colors.ravel())
    
    #inds = p.imag >= 0
    #p = p[inds]
    #r = r[inds]
    
    for i in range(p.shape[0]):
        print(p[i], r[i]/(2.0*np.pi))
    print(p.shape[0])
    
    t = np.linspace(0, 100, 10000)
    def Ct(p, r, t):
        inds = p.real <= 0
        p = p[inds]
        r = r[inds]
        P, T = np.meshgrid(p, t)
        return 1.0j*np.exp(P*T)@r
    
    plt.show()
    #plt.plot(t, np.real(1.0/(t+1.0j)/(2*np.pi)))
    plt.plot(t, np.real(Ct(p, r, t)))
    plt.plot(t, np.imag(Ct(p, r, t)))
    plt.plot(t, np.real(alpha*wc*wc/(1-1.0j*wc*t)**2))
    plt.show()
    
    #plt.plot(t, np.real(np.pi*alpha/2.0 * 200 *np.sqrt(np.pi)/np.power(1-20j*t, 1.5)), '--')
    #plt.plot(t, np.imag(np.pi*alpha/2.0 * 200 *np.sqrt(np.pi)/np.power(1-20j*t, 1.5)), '--')
    
    
    #plt.plot(t, np.real(Ct(poles2, res2, t))*2.0*np.pi)
    #plt.plot(t, np.imag(Ct(poles2, res2, t))*2.0*np.pi)
    #plt.plot(t, np.real(1.0/(8*np.pi)*wc*((-1.0+1.0j*t*wc)/(-wc*wc*tau*tau+(wc*t+1.0j)*(wc*t+1.0j) ) + 2.0j/(wc*t+1.0j))))
    
    #plt.plot(t, np.real(1.0/(8*np.pi)*wc*( (-1.0+1.0j*t*wc)/(-wc*wc*tau*tau*4+(wc*t+1.0j)*(wc*t+1.0j) ) + (-1.0+1.0j*t*wc)/(-wc*wc*tau*tau+(wc*t+1.0j)*(wc*t+1.0j) ) + 2.0j/(wc*t+1.0j))))
    #plt.plot(t, np.imag(1.0/(t+1.0j)/(2*np.pi)))
    #plt.plot(t, np.imag(Ct(p, r, t)))
        
    #plt.scatter(z.real, z.imag)# np.abs(func(Z)))
    #plt.plot(Zv, func1(Zv))
    #plt.plot(Zv, func3(Zv))

if __name__ == "__main__":
    test()
