import numpy as np
import matplotlib.pyplot as plt
import h5py 

def plot(fname, ax, style='o-'):
    try:
        h5 = h5py.File(fname, 'r')
        chis = np.array(h5.get('chis'))
        mean = np.array(h5.get('mean'))
        stderr = np.array(h5.get('stderr'))
        print(chis, mean, stderr)
        h5.close()
        ax.loglog(chis, mean, style, linewidth=3)
        ax.fill_between(chis, mean-2*stderr, mean+2*stderr, alpha=0.3)
        return chis[-1], mean[-1]
    except:
        return 0,0

chis = np.arange(2, 128)
fig, ax= plt.subplots(nrows=1, ncols=3, sharey=True)
plt.subplots_adjust(wspace=0, hspace=0)
mv = 0
chim, mm = plot("chi_scaling_mps_ts_subspace.h5", ax[0])
mv = max(mm, mv)
ax[0].loglog(chis, np.power(chis/chim, 3)*mm, 'k--', zorder=-1)
chim, mm = plot("chi_scaling_mps_subspace.h5", ax[0], style='o-')
mv = max(mm, mv)
ax[0].loglog(chis, np.power(chis/chim, 3)*mm, 'k--', zorder=-1)


chim, mm = plot("chi_scaling_binary_ts_subspace.h5", ax[1])
mv = max(mm, mv)
ax[1].loglog(chis, np.power(chis/chim, 5)*mm, 'k--', zorder=-1)
chim, mm = plot("chi_scaling_binary_subspace.h5", ax[1], style='o-')
mv = max(mm, mv)
ax[1].loglog(chis, np.power(chis/chim, 4)*mm, 'k--', zorder=-1)

chim, mm = plot("chi_scaling_ternary_ts_subspace.h5", ax[2], style='o-')
mv = max(mm, mv)
ax[2].loglog(chis, np.power(chis/chim, 7)*mm, 'k--', zorder=-1)

chim, mm = plot("chi_scaling_ternary_subspace.h5", ax[2], style='o-')
mv = max(mm, mv)
ax[2].loglog(chis, np.power(chis/chim, 5)*mm, 'k--', zorder=-1)
#ax[0].set_ylim([1e-2, mv*2])
ax[0].set_ylabel('Runtime (s)')
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none',which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$\chi$')
plt.show()
