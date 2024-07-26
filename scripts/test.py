import numpy as np
import sys
import time

sys.path.append("../../")
from pyttn import *



N=4
H2 = SOP(N+1)
sysinf = system_modes()
topo = ntree("(1(2(2))(2(3(8))(4(8)))(5(6(8))(7(8))))")
ntreeBuilder.sanitise(topo)
print(topo)
A = ttn(topo, dtype=np.complex128)
A.random()
print(A)

w = np.arange(N)
c = np.arange(N)
T = np.random.rand(N, N)
np.fill_diagonal(T, w)
sbg = spin_boson(0.0, 0.0, T, geom="generic")
#sbg = spin_boson(0, 0.0, 1.0, w, c, geom="star")
sbg.hamiltonian(H2)
print(H2)
sbg.mode_dims = [8 for i in range(N)]
sbg.hamiltonian(H2)
sbg.system_info(sysinf)
sop_op = sop_operator_complex(H2, A, sysinf)
print(sysinf)

print(A.ntensors())
si = [1 for x in range(A.nmodes())]
si[0] = 0
A.set_state(si)
print(si)
mel = matrix_element(A)
print(mel(A))
print(np.sum(w))
print(mel(sop_op, A))
sop_op = sop_operator_complex(H2, A, sysinf, compress=False)
print(mel(sop_op, A))
print(w, c)
print(np.sum(w*w), np.sum(w))






















np.random.seed(0)

N = 5
T = np.random.rand(N, N)
U = np.random.rand(N, N, N, N)
print(T, U)


es = electronic_structure(T, U)

H2 = SOP(N)
sysinf = system_modes(N)
es.hamiltonian(H2)
es.system_info(sysinf)
#H2.jordan_wigner(sysinf)

dims = [2 for i in range(N)]
topo = ntreeBuilder.mlmctdh_tree(dims, 2, 16)
ntreeBuilder.sanitise(topo)
A = ttn(topo, dtype=np.complex128)
A.random()
B = ttn(A)

#H2.jordan_wigner(sysinf)

def run_cSOP():
    sop_op = sop_operator_complex(H2, A, sysinf)
    mel = matrix_element(A)
    t = time.time()
    for i in range(10):
        print(mel(sop_op, A))
    mel.clear()
    sop_op.clear()
    print(time.time()-t)
    return time.time()-t

def run_SOP():
    sop_op = sop_operator_complex(H2, A, sysinf, compress=False)
    mel = matrix_element(A)
    t = time.time()
    for i in range(10):
        print(mel(sop_op, A))
    mel.clear()
    sop_op.clear()
    print(time.time()-t)
    return time.time()-t

print(run_cSOP())
print(run_SOP())


exit()

N = 4
H2 = SOP(N)

for i in range(N):
    for j in range(i,N):
        for k in range(N):
            for l in range(k,N):
                print(i, j, k, l)
                H2 += (i*j*k*l+1.0)*sOP("adag", i)*sOP("a", j) * sOP("adag", k) * sOP("a", l)



tree = ntree("(1(2(3(4))(4(4)))(5(6(4))(7(4))))")
tree.root().value = 15
print(tree.size())
print(tree.nleaves())

topo = ntree("(1(2(3(8))(4(8)))(5(6(8))(7(8))))")
ntreeBuilder.sanitise(topo)
print(topo)
A = ttn(topo, dtype=np.complex128)
A.random()

sysinf = system_modes(4)
for i in range(4):
    sysinf[i] = mode_data(8, mode_type.boson_mode)

sop_op = sop_operator_complex(H2, A, sysinf)

print(type(M))
op = site_operator(
    np.random.rand(8, 8), 
    dtype=np.complex128, 
    optype="matrix"
    )

A.set_orthogonality_centre(2)
print(A[2])
A.apply_one_body_operator(op, 0)
print(A[2])
A.apply_one_body_operator(op, 1)
print(A[2])
A.apply_one_body_operator(op, 2)
print(A[2])
A.apply_one_body_operator(op, 3)
print(A[2])
A.apply_one_body_operator(op, 0)
print(A[2])
A.apply_one_body_operator(op, 1)
print(A[2])
A.apply_one_body_operator(op, 2)
print(A[2])
A.apply_one_body_operator(op, 3)
print(A[2])

for i in A:
    print(i.data())

print(len(A))
print(A.ntensors())
A.set_state([1 for x in range(A.nmodes())])
mel = matrix_element(A)
mel(A)
print(mel(A))
print(mel([op, op, op, op], [0, 1, 2, 3], A))
print(mel(op, 0, A))
print(mel(op, 1, A))
print(mel(op, 2, A))
print(mel(op, 3, A))
print(mel(sop_op, A))
exit()
print(A)
A.clear()
print(H2)
T = np.random.rand(N, N)
U = np.random.rand(N, N, N, N)
aim = electronic_structure(T, U)

sbg = spin_boson(0.0, 1.0, T, geom="generic")
sbg.hamiltonian(H2)
print(H2)



sysinf = system_modes()
sbg = spin_boson(0.0, 1.0, np.random.rand(N), np.random.rand(N), geom="chain")
print(H2)

sbg = spin_boson(2, 0.0, 1.0, np.random.rand(N), np.random.rand(N), geom="star")
sbg.hamiltonian(H2)
sbg.mode_dims = [8 for i in range(N)]
sbg.hamiltonian(H2)
sbg.system_info(sysinf)

topo = ntree("(1(2(2))(2(3(8))(4(8)))(5(6(8))(7(8))))")
ntreeBuilder.sanitise(topo)
print(topo)
A = ttn(topo, dtype=np.complex128)
A.random()


sop_op = sop_operator_complex(H2, A, sysinf)

print(H2)
print(sysinf)





#op = ops.matrix_real(linalg.matrix_real(np.random.rand(2,2)))
#print(op)
#op = ops.matrix_real(np.random.rand(2,2))
#print(op)
#
#op = ops.direct_product_real([linalg.matrix_real(np.random.rand(2,2)), linalg.matrix_real(np.random.rand(3, 3)), linalg.matrix_real(np.random.rand(4,4))])
#print(op)
#op = ops.direct_product_real([np.random.rand(2,2), np.random.rand(3, 3), np.random.rand(4,4)])
#print(op)


