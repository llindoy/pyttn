import numpy as np
import matplotlib.pyplot as plt
import h5py 

def plot(fname, ax, style='o-'):
    h5 = h5py.File(fname, 'r')
    chis = np.array(h5.get('Nbs'))
    mean = np.array(h5.get('mean'))
    stderr = np.array(h5.get('stderr'))
    print(chis.shape, mean.shape)
    h5.close()
    ax.plot(chis[:mean.shape[0]], mean, style, linewidth=3)
    #ax.fill_between(chis, mean-2*stderr, mean+2*stderr, alpha=0.3)
    return chis[mean.shape[0]-1], mean[-1]

chis = np.arange(2, 512)
fig, ax= plt.subplots(nrows=2, ncols=2, sharey=False, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
mv = 0

chim, mm = plot("bath_size_scaling_mps_no_id.h5", ax[0,0])
ax[0,0].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)
mv = max(mm, mv)
chim, mm = plot("bath_size_scaling_mps_id.h5", ax[0,0])
mv = max(mm, mv)
ax[0,0].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)

chim, mm = plot("bath_size_scaling_binary_no_id.h5", ax[0,1])
ax[0,1].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)
mv = max(mm, mv)
chim, mm = plot("bath_size_scaling_binary_id.h5", ax[0,1])
mv = max(mm, mv)
ax[0,1].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)

chim, mm = plot("bath_size_scaling_ternary_no_id.h5", ax[1,0])
ax[1,0].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)
mv = max(mm, mv)
chim, mm = plot("bath_size_scaling_ternary_id.h5", ax[1,0])
mv = max(mm, mv)
ax[1,0].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)

chim, mm = plot("bath_size_scaling_quaternary_no_id.h5", ax[1,1])
ax[1,1].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)
mv = max(mm, mv)
chim, mm = plot("bath_size_scaling_quaternary_id.h5", ax[1,1])
mv = max(mm, mv)
ax[1,1].plot(chis, np.power(chis/chim, 1)*mm, 'k--', zorder=-1)

fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none',which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$N$')
plt.ylabel(r'Runtime (s)')
plt.show()
