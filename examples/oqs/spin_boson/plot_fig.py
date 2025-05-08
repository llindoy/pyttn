# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

plt.rcParams.update({'font.size':16})
def plot():
    pars = []
    t = None

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(2.25*1.5, 2.25*2.5))
    fname = "sbm_heom.h5"
    try:
        h5 = None
        h5 = h5py.File(fname, "r")
        t = np.array(h5.get("t"))
        Sz = np.array(h5.get("Sz"))
        norm = np.array(h5.get("norm"))

        ax[0].plot(
            t, np.real(Sz) , "-", label="HEOM",linewidth=3)
        ax[1].semilogy(
            t, np.real(norm) , "-", label="HEOM",linewidth=3        )
        h5.close()
    except:
        print("Failed to read input file")

    fname = "sbm_pseudomode.h5"
    try:
        h5 = None
        h5 = h5py.File(fname, "r")
        t = np.array(h5.get("t"))
        Sz = np.array(h5.get("Sz"))
        norm = np.array(h5.get("norm"))

        ax[0].plot(
            t, np.real(Sz) , "-", label="HEOM",linewidth=3        )
        ax[1].semilogy(
            t, np.real(norm) , "-", label="Pseudomode",linewidth=3        )
        h5.close()
    except:
        print("Failed to read input file")

    plt.subplots_adjust(hspace=0.)
    ax[0].set_ylim([0.995, 1])
    ax[0].set_xlim([0, 0.4])
    ax[1].set_xlim([0, 0.4])
    ax[0].set_xlabel(r'$\Delta t$')
    ax[1].set_xlabel(r'$\Delta t$')
    ax[0].set_ylabel(r'$\langle \hat{\sigma}_z(t) \rangle$')
    ax[1].set_xlabel(r'$\Delta t$')
    ax[1].set_ylabel(r'$| \hat{\rho}(t) \rangle $')
    #ax[1].legend(frameon=False)
    plt.savefig('out.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot density matrix elements output by a heom calculation."
    )
    plot()
