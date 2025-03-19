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

def bath_size_scaling_SOP_vs_hSOP():
    chi = 20
    nbose = 10
    Nbs = [2, 4, 8, 16, 32, 64, 128, 256]
    timings_hSOP = []
    stdevs_hSOP = []

    timings_SOP = []
    stdevs_SOP = []
    nstep = 20
    #run through all steps sizes and compute the cost compressing the SOP
    for nb in Nbs:
        m, std = sbm_dynamics_timing(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, compress = True)
        timings_hSOP.append(m)
        stdevs_hSOP.append(std/np.sqrt(nstep))
        print(nb, m, std/np.sqrt(1.0*nstep))

    h5 = h5py.File("bath_size_scaling_hSOP.h5", 'w')
    h5.create_dataset('Nbs', data=np.array(Nbs))
    h5.create_dataset('mean', data=np.array(timings_hSOP))
    h5.create_dataset('stderr', data=np.array(stdevs_hSOP))
    h5.close()

    #run through all steps sizes and compute the cost without compressing the SOP
    for nb in Nbs:
        m, std = sbm_dynamics_timing(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, compress = False)
        timings_SOP.append(m)
        stdevs_SOP.append(std/np.sqrt(nstep))
        print(nb, m, std/np.sqrt(1.0*nstep))

    h5 = h5py.File("bath_size_scaling_SOP.h5", 'w')
    h5.create_dataset('Nbs', data=np.array(Nbs))
    h5.create_dataset('mean', data=np.array(timings_SOP))
    h5.create_dataset('stderr', data=np.array(stdevs_SOP))
    h5.close()

bath_size_scaling_SOP_vs_hSOP()
