# This files is part of the pyTTN package.
#(C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

import pyttn
from pyttn import oqs, utils
from pyttn.ttns.sop import OperatorBuilder, SystemInfo
from pyttn.utils import visualise_tree
from xy_tree import cayley_edges


def visualise(Nl, alpha=0.32, wc=4, eta=0.04, K=4, L=20, Lmin=4):
    """Visualise the joint system+bath tree topology that :class:`~pyttn.oqs.MethodBuilder` proposes automatically for a Cayley-tree XY model - replacing ``cayley_helper.build_system_topology`` entirely."""

    @jit(nopython=True)
    def J(w):
        return 2 * np.pi * alpha * w * np.exp(-np.abs(w / wc) ** 2)

    edges, Ns = cayley_edges(Nl, d=3)
    labels = [f"spin{i}" for i in range(Ns)]
    sysinfo = SystemInfo()
    for label in labels:
        sysinfo[label] = pyttn.tls_mode()

    b = OperatorBuilder()
    Hsys = b.op("sz", labels[0])
    for i in range(1, Ns):
        Hsys = Hsys + b.op("sz", labels[i])
    for i, j in edges:
        Hsys = Hsys + (1 - eta) * b.op("sx", labels[i]) * b.op("sx", labels[j]) + (1 + eta) * b.op("sy", labels[i]) * b.op("sy", labels[j])
    model = oqs.OQSModel(system_info=sysinfo, system_generator=b.wrap(Hsys))

    for i in range(Ns):
        bath = oqs.BosonicBath(J, wmax=wc * 10)
        coupling = OperatorBuilder()
        params = {"decomposition": oqs.ESPRITDecomposition(K=K, tmax=10.0, Nt=100), "truncation": utils.EnergyTruncation(10 * wc, Lmax=L, Lmin=Lmin), "chi0": 8, "chi": 16}
        model.add_bath(bath, coupling.wrap(coupling.op("sz", labels[i])), tag=f"bath{i}", params=params)

    result = oqs.MethodBuilder(model).build("heom", min_chi=8, max_chi=16)
    visualise_tree(result.topology.tree, prog="twopi", add_labels=False)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise the automatically proposed tree topology for a dissipative XY Cayley tree model")
    parser.add_argument("--Nl", type=int, default=3)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--Lmin", type=int, default=4)

    args = parser.parse_args()
    visualise(args.Nl, K=args.K, L=args.L, Lmin=args.Lmin)
