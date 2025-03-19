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
import h5py

from sbm_timing_helper import sbm_dynamics_timing


def chi_scaling_mps_binary_ternary(adaptive=False):
    nbose = 10
    nb = 32
    nstep = 20

    jtype = ""
    if adaptive:
        jtype = "_subspace"

    chis = [2, 4, 8, 16, 32, 48, 64]  # , 96, 128, 160, 192, 256, 384]
    timings_mps = []
    stdevs_mps = []

    chivs = []
    for chi in chis:
        m, std = sbm_dynamics_timing(
            nb,
            1.0,
            5,
            1.0,
            0.0,
            1.0,
            chi,
            nbose,
            0.01,
            nstep=nstep,
            degree=1,
            compress=True,
            adaptive=adaptive,
        )
        print(chi, m, std / np.sqrt(1.0 * nstep))
        chivs.append(chi)
        timings_mps.append(m)
        stdevs_mps.append(std / np.sqrt(1.0 * nstep))
        h5 = h5py.File("chi_scaling_mps" + jtype + ".h5", "w")
        h5.create_dataset("chis", data=np.array(chivs))
        h5.create_dataset("mean", data=np.array(timings_mps))
        h5.create_dataset("stderr", data=np.array(stdevs_mps))
        h5.close()

    timings_binary = []
    stdevs_binary = []
    chivs = []

    chis = [2, 4, 8, 16, 32]  # , 48, 64, 96, 128, 160, 192]
    for chi in chis:
        m, std = sbm_dynamics_timing(
            nb,
            1.0,
            5,
            1.0,
            0.0,
            1.0,
            chi,
            nbose,
            0.01,
            nstep=nstep,
            degree=2,
            compress=True,
            adaptive=adaptive,
        )
        chivs.append(chi)
        timings_binary.append(m)
        stdevs_binary.append(std / np.sqrt(1.0 * nstep))
        print(chi, m, std / np.sqrt(1.0 * nstep))

        h5 = h5py.File("chi_scaling_binary" + jtype + ".h5", "w")
        h5.create_dataset("chis", data=np.array(chivs))
        h5.create_dataset("mean", data=np.array(timings_binary))
        h5.create_dataset("stderr", data=np.array(stdevs_binary))
        h5.close()

    nb = 27
    chis = [2, 4, 8, 16]  # , 24, 32, 40, 48, 56]
    timings_3 = []
    stdevs_3 = []
    chivs = []
    for chi in chis:
        m, std = sbm_dynamics_timing(
            nb,
            1.0,
            5,
            1.0,
            0.0,
            1.0,
            chi,
            nbose,
            0.01,
            nstep=nstep,
            degree=3,
            compress=True,
            adaptive=adaptive,
        )
        chivs.append(chi)
        timings_3.append(m)
        stdevs_3.append(std / np.sqrt(1.0 * nstep))

        print(chi, m, std / np.sqrt(1.0 * nstep))

        h5 = h5py.File("chi_scaling_ternary" + jtype + ".h5", "w")
        h5.create_dataset("chis", data=np.array(chivs))
        h5.create_dataset("mean", data=np.array(timings_3))
        h5.create_dataset("stderr", data=np.array(stdevs_3))
        h5.close()


chi_scaling_mps_binary_ternary(adaptive=False)
