N = 3

sysinf = system_modes(N)
for i in range(N):
    sysinf[i] = qubit_mode()

w_0 = 1
w = 2
Rabi_freq = 5
t0 = 0

H = SOP(N)

# Add time-dependent couplings in bath using func_class, defined above
for i in range(N):
    H += w_0/2*sOP("z",i)
    H += Rabi_freq*coeff(lambda t : np.cos(w*t))*sOP("x", i) 

print(H)
