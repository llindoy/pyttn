import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy as sp

import sys
sys.path.append("../../")
from pyttn import *
from pyttn import oqs
from pyttn.utils import visualise_tree



dt = 0.02
nstep = 100

beta = None
W = 10
Gamma = 1
def V(w):
    return np.where(np.abs(w) <= W, Gamma*np.sqrt(1-(w*w)/(W*W)), 0.0)

bath = oqs.FermionicBath(V, beta=beta)
dkf, zkf = bath.expfit(oqs.ESPRITDecomposition(K=4, tmax=nstep*dt, Nt = nstep), sigma = '+')
dke, zke = bath.expfit(oqs.ESPRITDecomposition(K=4, tmax=nstep*dt, Nt = nstep), sigma = '-')

print(zkf)
print(zke)
exit()

import matplotlib.pyplot as plt
t = np.linspace(0, dt*nstep, nstep)
# plt.ion()

plt.plot(t, bath.Ct(t, sigma='+'))
plt.plot(t, oqs.FermionicBath.Ctexp(t, dkf, 1.0j*zkf, sigma = '+'), 'bo', label = 'Bath correlation function')
plt.plot(t, np.imag(bath.Ct(t, sigma='+')))
plt.plot(t, np.imag(oqs.FermionicBath.Ctexp(t, dkf, 1.0j*zkf, sigma = '+')), 'bo', label = 'Bath correlation function')
# plt.plot(t, oqs.FermionicBath.Ctexp(t, dkf, 1.0j*zkf, sigma = '+'), label = '+')
# plt.plot(t, oqs.FermionicBath.Ctexp(t, dke, -1.0j*zke, sigma = '-'), label = '-')
plt.plot(t, bath.Ct(t, sigma='-'))
plt.plot(t, oqs.FermionicBath.Ctexp(t, dke, -1.0j*zke, sigma = '-'), 'ro', label = 'Discretisation')
plt.plot(t, np.imag(bath.Ct(t, sigma='-')))
plt.plot(t, np.imag(oqs.FermionicBath.Ctexp(t, dke, -1.0j*zke, sigma = '-')), 'ro', label = 'Discretisation')
plt.xlabel("Time")
plt.ylabel("Bath correlation function")
plt.legend()
plt.show()


Nbo = dkf.shape[0]
Nbe = dke.shape[0]
Nb = Nbo + Nbe
N = Nb + 1



chi = 16
chiU = 48

topo = ntree(str("(1(chi(4(4))(chi))(chi(4(4))(chi)))").replace('chi', str(chi)))
ntreeBuilder.mps_subtree(topo()[0][1], [2*2 for _ in range(Nbo)], chi)
ntreeBuilder.mps_subtree(topo()[0][1], [2*2 for _ in range(Nbe)], chi)
ntreeBuilder.mps_subtree(topo()[1][1], [2*2 for _ in range(Nbo)], chi)
ntreeBuilder.mps_subtree(topo()[1][1], [2*2 for _ in range(Nbe)], chi)
ntreeBuilder.sanitise(topo)

capacity = ntree(str("(1(chiU(4(4))(chiU))(chiU(4(4))(chiU)))").replace('chiU', str(chiU)))
ntreeBuilder.mps_subtree(capacity()[0][1], [2*2 for _ in range(Nbo)], chiU)
ntreeBuilder.mps_subtree(capacity()[0][1], [2*2 for _ in range(Nbe)], chiU)
ntreeBuilder.mps_subtree(capacity()[1][1], [2*2 for _ in range(Nbo)], chiU)
ntreeBuilder.mps_subtree(capacity()[1][1], [2*2 for _ in range(Nbe)], chiU)
ntreeBuilder.sanitise(capacity)

print(dkf.shape, dkf)
print(N)

visualise_tree(topo)
plt.show()
print(topo)
print(capacity)


mode_ordering = [N - (x+1) for x in range(N)] + [N + x for x in range(N)]

modes_f_d = [N-1 - (x+1) for x in range(Nbo)]
modes_f_u = [N+1 + x for x in range(Nbo)]
modes_f = modes_f_d + modes_f_u

modes_e_d = [N-1-Nbo - (x+1) for x in range(Nbe)]
modes_e_u = [N+1+Nbo + x for x in range(Nbe)]
modes_e = modes_e_d + modes_e_u

print(Nbo, Nbe)
print(2*N)
print(N)
print(mode_ordering)
print(modes_f_d)
print(modes_f_u)
print(modes_f)
print(modes_e_d)
print(modes_e_u)
print(modes_e)



sysinf = system_modes(2*N)
sysinf.mode_indices = mode_ordering
for i in range(2*N):
    sysinf[i] = [fermion_mode(), fermion_mode()]



gkf = np.real(zkf)
Ekf = np.imag(zkf)
Vkf = np.real(np.sqrt(dkf))
Mkf = -np.imag(np.sqrt(dkf))

gke = np.real(zke)
Eke = np.imag(zke)
Vke = np.real(np.sqrt(dke))
Mke = -np.imag(np.sqrt(dke))

eps = -1.25*np.pi
U = 2.5*np.pi
# U = 0


H = SOP(4*N)


def fop(label, mode, ltype=None):
    if ltype == '~':
        return fOP(label, 2*mode+1)
    else:
        return fOP(label, 2*mode)

# System Liouvillian
H += eps*fop("n", (N-1))
H += eps*fop("n", (N))
H += U*fop("n", (N-1))*fop("n",(N))

H += -eps*fop("n", (N-1), '~')
H += -eps*fop("n", N, '~')
H += -U*fop("n", N-1, '~')*fop("n",N, '~')



def add_bath_full(H, Ek, Vk, gk, Mk, modes, Nu):
    for i in range(len(modes)):
        # Occupied subchain, spin up
        # # Bath Liouvillian
        H += Ek[i]*fop("n", modes[i])
        H += -Ek[i]*fop("n", modes[i], '~')
        
        # # System - bath Liouvillian
        H += Vk[i]*fop("cdag", (N))*fop("a", modes[i])
        H += -Vk[i]*fop("adag", modes[i],'~')*fop("c", (N),'~')      #

        H += Vk[i]*fop("adag", modes[i])*fop("c", (N))
        H += -Vk[i]*fop("cdag", (N), '~')*fop("a", modes[i], '~')    #
            
        # Bath dissipator
        H +=2.0j*gk[i]*fop("a", modes[i])*fop("a", modes[i],'~')     # # 
        H +=-1.0j*gk[i]*fop("adag", modes[i])*fop("a", modes[i])
        H +=-1.0j*gk[i]*fop("adag", modes[i],'~')*fop("a", modes[i],'~')
        #
        # # System - bath dissipator
        coeff = 2.0j
        if N > modes[i]:
            coeff = -2.0j
        H +=coeff*Mk[i]*fop("c", (N))*fop("a", modes[i],'~')      # #

        coeff = 2.0j
        if N < modes[i]:
            coeff = -2.0j
        H += coeff*Mk[i]*fop("a", modes[i])*fop("c", (N),'~')      # #

        H +=-1.0j*Mk[i]*fop("adag", modes[i])*fop("c", (N))
        H +=-1.0j*Mk[i]*fop("cdag", (N))*fop("a", modes[i])
        H +=-1.0j*Mk[i]*fop("cdag", (N),'~')*fop("a", modes[i],'~')   #
        H +=-1.0j*Mk[i]*fop("adag", modes[i],'~')*fop("c", (N),'~')   #

    return H

def add_bath_empty(H, Ek, Vk, gk, Mk, modes, N):
    for i in range(len(modes)):
        # Occupied subchain, spin up
        # # Bath Liouvillian
        H += Ek[i]*fop("n", modes[i])
        H += -Ek[i]*fop("n", modes[i], '~')
        
        # # System - bath Liouvillian
        H += Vk[i]*fop("cdag", (N))*fop("a", modes[i])
        H += -Vk[i]*fop("adag", modes[i],'~')*fop("c", (N),'~')      #

        H += Vk[i]*fop("adag", modes[i])*fop("c", (N))
        H += -Vk[i]*fop("cdag", (N), '~')*fop("a", modes[i], '~')    #

        # # Bath dissipator
        H +=2.0j*gk[i]*fop("adag", modes[i])*fop("adag", modes[i],'~')     # #
        H +=-1.0j*gk[i]*fop("a", modes[i])*fop("adag", modes[i])
        H +=-1.0j*gk[i]*fop("a", modes[i],'~')*fop("adag", modes[i],'~')
        #
        # # System - bath dissipator
        coeff = 2.0j
        if N > modes[i]:
            coeff = -2.0j
        H += coeff*Mk[i]*fop("cdag", (N))*fop("adag", modes[i],'~')      # #

        coeff = 2.0j
        if N < modes[i]:
            coeff = -2.0j
        H += coeff*Mk[i]*fop("adag", modes[i])*fop("cdag", (N),'~')      # #

        H +=-1.0j*Mk[i]*fop("a", modes[i])*fop("cdag", (N))
        H +=-1.0j*Mk[i]*fop("c", (N))*fop("adag", modes[i])
        H +=-1.0j*Mk[i]*fop("c", (N),'~')*fop("adag", modes[i],'~')   #
        H +=-1.0j*Mk[i]*fop("a", modes[i],'~')*fop("cdag", (N),'~')   #
    return H

H = add_bath_full(H, Ekf, Vkf, gkf, Mkf, modes_f_d, N-1)
H = add_bath_full(H, Ekf, Vkf, gkf, Mkf, modes_f_u, N)

H = add_bath_empty(H, Eke, Vke, gke, Mke, modes_e_d, N-1)
H = add_bath_empty(H, Eke, Vke, gke, Mke, modes_e_u, N)

print(H)

H.jordan_wigner(sysinf)

print(H)

A = ttn(topo, capacity)

state = [0 for _ in range(2*N)]
for i in range(Nbo):
    state[1+i] = 0b11
    state[N+1+i] = 0b11


print(state)
A.set_state(state)

h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)



obstree = ntree(str("(1(chi(4(4))(chi))(chi(4(4))(chi)))").replace('chi', str(chi)))
ntreeBuilder.mps_subtree(obstree()[0][1], [2*2 for _ in range(Nbo)], 1, 1)
ntreeBuilder.mps_subtree(obstree()[0][1], [2*2 for _ in range(Nbe)], 1, 1)
ntreeBuilder.mps_subtree(obstree()[1][1], [2*2 for _ in range(Nbo)], 1, 1)
ntreeBuilder.mps_subtree(obstree()[1][1], [2*2 for _ in range(Nbe)], 1, 1)
ntreeBuilder.sanitise(obstree)

# visualise_tree(obstree)

id_ttn = ttn(obstree)
prod_state = []
for i in range(2*N):
    state_vec = np.identity(2, dtype=np.complex128).flatten()
    prod_state.append(state_vec)
id_ttn.set_product(prod_state)

# n_e = site_operator(fOP("n", 2*(N)), sysinf)
n_e = SOP(4*N)
n_e += fop("n", (N))
# n_e += fOP("n", (N),'~')
n_e += fop("n", (N-1))
# n_e += fOP("n", (N-1),'~')
n_e = sop_operator(n_e, A, sysinf)

mel = matrix_element(A)



sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace', subspace_neigs = 6, subspace_krylov_dim = 12)
sweep.spawning_threshold = 1e-4
sweep.unoccupied_threshold = 1e-4
sweep.minimum_unoccupied = 0

sweep.coefficient = -1.0j

res = np.zeros(nstep+1)
maxchi = np.zeros(nstep+1)
renorm = np.zeros(nstep+1)
res[0] = np.real(mel(n_e, A, id_ttn))
maxchi[0] = A.maximum_bond_dimension()
renorm[0] = np.real(mel(id_ttn, A))

i=0
print((i)*dt, res[i], renorm[i], maxchi[i], np.real(mel(A, A)))

tp = 0
ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
for i in range(5):
    dti = ts[i]-tp
    print(ts, dt)
    sweep.dt = dti
    sweep.step(A, h)
    tp = ts[i]
    
res[1] = np.real(mel(n_e, A, id_ttn))
maxchi[1] = A.maximum_bond_dimension()
renorm[1] = np.real(mel(id_ttn, A))

i=1
print((i)*dt, res[i], renorm[i], maxchi[i], np.real(mel(A, A)))

import time
import sys
import h5py
ofname = "siam_pf_fixW.h5"

sweep.dt = dt
for i in range(1,nstep):
    t1 = time.time()
    sweep.step(A, h)
    t2 = time.time()
    renorm[i+1] = np.real(mel(id_ttn, A))
    # A*=(1/renorm)
    res[i+1] = np.real(mel(n_e, A, id_ttn))
    maxchi[i+1] = A.maximum_bond_dimension()

    print((i+1)*dt, res[i+1], renorm[i+1], maxchi[i+1], np.real(mel(A, A)))
    sys.stdout.flush()
    if(i % 10 == 0):
        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
        h5.create_dataset('n_u', data=res)
        h5.create_dataset('maxchi', data=maxchi)
        h5.create_dataset('renorm', data=renorm)
        h5.close()

h5 = h5py.File(ofname, 'w')
h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
h5.create_dataset('n_u', data=res)
h5.create_dataset('maxchi', data=maxchi)
h5.create_dataset('renorm', data=renorm)
h5.close()

# plt.ioff()
