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


def plot(fnames):
    for fname in fnames:
        pars = []
        t = None
        try:
            h5 = None
            h5 = h5py.File(fname, "r")
            t = np.array(h5.get("t"))
            N = len(h5.keys()) - 2
            res = np.zeros((N, len(t)))
            for i in range(N):
                label = "|" + str(i) + "><" + str(i) + "|"
                res[i, :] = np.array(h5.get(label))

            h5.close()
        except:
            print("Failed to read input file")
            continue

        for i in range(len(t)):
            res[:, i] = res[:, i] / np.max(res[:, i])

        plt.figure(fname)
        plt.imshow(res)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot density matrix elements output by a heom calculation."
    )
    parser.add_argument("fname", nargs="+")

    args = parser.parse_args()
    plot(args.fname)
