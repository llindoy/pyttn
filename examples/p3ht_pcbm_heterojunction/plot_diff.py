import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

def plot(fnames):
    pars = []
    t = None
    print(fnames)
    for fname in fnames:
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))
            pars.append(np.array(h5.get("|CS9><CS9|")))

            h5.close()
        except:
            print("Failed to read input file")
            continue

    for i in range(1, len(pars)):
        try:
            plt.plot(t, 100*(pars[i]-pars[0]), '-', linewidth=3)
        except:
            print("Failed to plot: ")
    plt.xlim([0, 200])


if __name__ == "__main__":
    plt.figure(1)
    plot(["model_B/out_mps_96_smaller_basis.h5", "model_B/out_mps_64.h5", "model_B/out_mps_48.h5", "model_B/out_mps_32.h5", "model_B/out_mps_24.h5", "model_B/out_mps_16.h5", "model_B/out_mps_8.h5"])
    plt.figure(2)
    plot(["model_B/out_mlmctdh_64.h5", "model_B/out_mlmctdh_48.h5", "model_B/out_mlmctdh_32.h5", "model_B/out_mlmctdh_24.h5", "model_B/out_mlmctdh_16.h5", "model_B/out_mlmctdh_8.h5"])
    plt.show()


