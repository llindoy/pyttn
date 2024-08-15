import numpy as np
import matplotlib.pyplot as plt
import h5py 

def plot(fname, ax):
    h5 = h5py.File(fname, 'r')
    t = np.array(h5.get('t'))
    Sz = np.array(h5.get('Sz'))
    runtime = np.array(h5.get('runtime'))
    ax[0].plot(t, Sz)
    ax[1].plot(t, runtime)

chis = np.arange(2, 384)
fig, ax= plt.subplots(nrows=1, ncols=2)
#plot("sbm_subspace_expansion_24_1e-05_0.0001.h5",ax)
plot("sbm_subspace_expansion_24_1e-06_0.0001.h5",ax)
plt.show()
