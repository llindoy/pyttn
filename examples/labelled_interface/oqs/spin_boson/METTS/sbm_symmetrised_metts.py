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

import copy
import functools

import h5py
import numpy as np

import pyttn
from pyttn.oqs.bath_fitting import softmspace
from pyttn.simulation import Ensemble, ResultsBuffer
from sbm_metts import build_system, evolve_imaginary_time, measurement_basis_operators, thermalize_and_checkpoint


def evolve_imaginary_time_symmetrised(A, B, h, sweep, sweepB, betasteps):
    """Symmetrised imaginary time evolution splitting the thermalisation between A and B (each renormalised as they evolve)."""
    beta_p = 0
    for bi in betasteps:
        sweep.dt = sweepB.dt = bi - beta_p
        beta_p = bi
        sweep.step(A, h)
        sweepB.step(B, h)
        rhoB = B.normalise()
    return rhoB


def _run_sample(index, result, checkpoint_file, beta, nbeta, dt, nstep, spawning_threshold, unoccupied_threshold, nunoccupied):
    np.random.seed(index)
    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    A.load(checkpoint_file)
    hb = pyttn.sop_operator(result.generator, A, result.system_modes)
    sweep = pyttn.tdvp(A, hb, krylov_dim=12, expansion="subspace")
    sweep.spawning_threshold, sweep.unoccupied_threshold, sweep.minimum_unoccupied, sweep.coefficient = spawning_threshold, unoccupied_threshold, nunoccupied, -1.0

    Uproj = measurement_basis_operators(result)
    beta_steps = softmspace(1e-6, beta / 2.0, nbeta)
    for _ in range(2):
        for collapse in (lambda: A.collapse_basis(Uproj, nchi=2), lambda: A.collapse(nchi=2)):
            collapse()
            A.normalise()
            sweep.prepare_environment(A, hb)
            evolve_imaginary_time(A, hb, sweep, beta_steps)

    op = pyttn.site_operator(pyttn.sOP("sz", result.site_map["spin"]), result.system_modes)
    B = copy.deepcopy(A)
    B.apply_one_body_operator(op)

    sweepB = pyttn.tdvp(B, hb, krylov_dim=12, expansion="subspace")
    sweepB.spawning_threshold, sweepB.unoccupied_threshold, sweepB.minimum_unoccupied, sweepB.coefficient = spawning_threshold, unoccupied_threshold, nunoccupied, -1.0
    sweep.prepare_environment(A, hb)
    sweepB.prepare_environment(B, hb)
    coeff = evolve_imaginary_time_symmetrised(A, B, hb, sweep, sweepB, beta_steps)

    h = pyttn.sop_operator(result.generator, A, result.system_modes)
    sweepA, sweepB = pyttn.tdvp(A, h, krylov_dim=12), pyttn.tdvp(B, h, krylov_dim=12)
    for s in (sweepA, sweepB):
        s.dt, s.coefficient = dt, -1.0j

    mel = pyttn.matrix_element(A, B)
    results = ResultsBuffer(["Sz"], nstep + 1, dtype=np.complex128)
    results.record(0, 0.0, {"Sz": mel(op, A, B) * coeff}, maxchi=A.maximum_bond_dimension())
    for i in range(nstep):
        sweepA.step(A, h)
        sweepB.step(B, h)
        results.record(i + 1, (i + 1) * dt, {"Sz": mel(op, B, A) * coeff}, maxchi=A.maximum_bond_dimension())
    return results


class METTSSample:
    """An :class:`~pyttn.simulation.Ensemble`-compatible sample built entirely from plain picklable arguments, so samples can safely run in worker processes."""

    def __init__(self, index, build_kwargs, checkpoint_file, beta, nbeta, dt, nstep, spawning_threshold, unoccupied_threshold, nunoccupied):
        self.index, self.build_kwargs, self.checkpoint_file = index, build_kwargs, checkpoint_file
        self.beta, self.nbeta, self.dt, self.nstep = beta, nbeta, dt, nstep
        self.spawning_threshold, self.unoccupied_threshold, self.nunoccupied = spawning_threshold, unoccupied_threshold, nunoccupied

    def run(self):
        result = build_system(**self.build_kwargs)
        self.results = _run_sample(self.index, result, self.checkpoint_file, self.beta, self.nbeta, self.dt, self.nstep, self.spawning_threshold, self.unoccupied_threshold, self.nunoccupied)
        return self.results


def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta=5, nbeta=100, nsamples=256, nstep=1, degree=1, ofname="sbm_thermal.h5", spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, nwarmup=4, n_workers=1):
    build_kwargs = {"Nb": Nb, "alpha": alpha, "wc": wc, "s": s, "eps": eps, "delta": delta, "chi": chi, "nbose": nbose, "degree": degree}
    result = build_system(**build_kwargs)

    checkpoint_file = ofname + ".checkpoint"
    thermalize_and_checkpoint(result, beta, nbeta, nwarmup, checkpoint_file, spawning_threshold, unoccupied_threshold, nunoccupied)

    sample_fn = functools.partial(METTSSample, build_kwargs=build_kwargs, checkpoint_file=checkpoint_file, beta=beta, nbeta=nbeta, dt=dt, nstep=nstep, spawning_threshold=spawning_threshold, unoccupied_threshold=unoccupied_threshold, nunoccupied=nunoccupied)
    samples = Ensemble(sample_fn, n_samples=nsamples, n_workers=n_workers).run()

    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=samples[0]["t"])
    h5.create_dataset("Sz", data=np.array([s["Sz"] for s in samples]))
    h5.close()
    os.remove(checkpoint_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Symmetrised METTS thermal dynamics of the zero temperature spin boson model")

    parser.add_argument("alpha", type=float)
    parser.add_argument("--wc", type=float, default=5)
    parser.add_argument("--s", type=float, default=1)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0)
    parser.add_argument("--delta", type=float, default=1)
    parser.add_argument("--beta", type=float, default=5)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--nbose", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--tmax", type=float, default=10)
    parser.add_argument("--fname", type=str, default="sbm_thermal.h5")
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--nbeta", type=int, default=100)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-5)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()
    nstep = int(args.tmax / args.dt)

    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta=args.beta, nstep=nstep, ofname=args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold=args.unoccupied_threshold, degree=args.degree, nsamples=args.nsamples, nbeta=args.nbeta, n_workers=args.workers)
