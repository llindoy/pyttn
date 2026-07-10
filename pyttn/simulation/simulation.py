# This files is part of the pyTTN package.
# (C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from pyttn import dmrg, matrix_element, sop_operator, tdvp
from pyttn.ttns.sop import lCSOP, lSOP

from .observable import Observable
from .results import ResultsBuffer


def _compiled_observable(obs: Observable, A, system_modes, site_map: Optional[Dict[str, int]]) -> Observable:
    """Compile a labelled observable operator if required. Returns ``obs`` unchanged unless its operator is an ``lSOP``/``lCSOP``, in which case it is compiled against ``site_map`` and ``system_modes``.
    """
    if not isinstance(obs.op, (lSOP, lCSOP)):
        return obs
    if site_map is None or system_modes is None:
        raise ValueError(f"Observable '{obs.label}' has a labelled (lSOP/lCSOP) operator, but no site_map/system_modes was given to compile it - pass site_map=... and system_modes=... (e.g. result.site_map/result.system_modes from a MethodBuilder.build(...) call) to the Simulation constructor.")
    compiled = sop_operator(obs.op.compile(site_map, len(site_map)), A, system_modes)
    return Observable(obs.label, op=compiled, mode=obs.mode)


def _normalise_states(states) -> tuple:
    """Convert ``states`` into a tuple of reference states."""
    if states is None:
        return ()
    if isinstance(states, (list, tuple)):
        return tuple(states)
    return (states,)


def _logarithmic_ramp(integrator, A, H, dt: float, t0: float = 0.0, nstep: int = 5, nscale: float = 1e-5) -> float:
    """Advance the first timestep using logarithmically spaced substeps.

    This can improve stability when handling weakly occupied modes at the start of a simulation.


    :param integrator: The integrator to advance 
    :type integrator: TDVP
    :param A: The state to evolve
    :type A: ttn or ms_ttn
    :param H: The generator
    :type H: sop_operator or ms_sop_operator
    :param dt: The timestep this ramp replaces
    :type dt: float
    :param t0: Initial time, defaults to 0.0
    :type t0: float, optional
    :param nstep: The number of logarithmically spaced substeps, defaults to 5
    :type nstep: int, optional
    :param nscale: The size of the first substep, as a fraction of ``dt``, defaults to 1e-5
    :type nscale: float, optional
    :return: The time reached after the ramp, ``t0 + dt``
    :rtype: float
    """
    tp = 0.0
    ts = np.logspace(np.log10(dt * nscale), np.log10(dt), nstep)
    for ti in ts:
        integrator.dt = ti - tp
        integrator.step(A, H)
        tp = ti
    return t0 + tp


class Simulation(ABC):
    """Base class for simulation workflows.
    
    Wraps a state, generator, integrator, observables, and results buffer into a common execution interface. Subclasses implement specific algorithms such as TDVP time evolution or DMRG optimisation.

    :param A: The state to act on
    :type A: ttn or ms_ttn
    :param H: The generator
    :type H: sop_operator or ms_sop_operator
    :param integrator: Optional pre-built integrator.
    :type integrator: TDVP or DMRG, optional
    :param expansion: The bond dimension expansion strategy to use
    :type expansion: str, optional
    :param integrator_kwargs: Additional integrator arguments.
    :type integrator_kwargs: dict, optional
    :param mel: Optional matrix-element evaluator.
    :type mel: matrix_element, optional
    :param observables: The observables to measure at each recorded step
    :type observables: list[Observable], optional
    :param reference_states: Additional fixed states passed to observables
    :type reference_states: object or sequence, optional
    :param system_modes: The system_modes describing ``A`` - only needed if any ``observables`` use a labelled (``lSOP``/``lCSOP``) operator
    :type system_modes: system_modes, optional
    :param site_map: Mapping from site label to physical mode index, required  alongside ``system_modes`` 
    :type site_map: dict[str, int], optional
    :param nrecords: The number of records the results buffer should have space for
    :type nrecords: int
    :param dtype: The dtype used for the observable columns of the results buffer
    :type dtype: type, optional
    """

    def __init__(self, A, H, integrator=None, expansion: str = "onesite", integrator_kwargs: Optional[dict] = None, 
                 mel=None, observables: Optional[Sequence[Observable]] = None, reference_states: Optional[Sequence] = None, 
                 system_modes=None, site_map: Optional[Dict[str, int]] = None, extra_labels: Optional[Sequence[str]] = None, 
                 nrecords: int = 1, dtype=np.complex128,
    ):
        self.A = A
        self.H = H
        self.integrator = integrator if integrator is not None else self._build_integrator(A, H, expansion, **(integrator_kwargs or {}))
        self.mel = mel if mel is not None else matrix_element(A)
        self.observables: List[Observable] = [_compiled_observable(obs, A, system_modes, site_map) for obs in (observables or [])]
        self.reference_states = _normalise_states(reference_states)

        labels = [obs.label for obs in self.observables] + list(extra_labels or [])
        self.results = ResultsBuffer(labels, nrecords, dtype=dtype)

    @staticmethod
    @abstractmethod
    def _build_integrator(A, H, expansion: str, **kwargs):
        """Build the default integrator for this simulation type."""

    @property
    def state(self):
        """The current state acted on by this simulation."""
        return self.A

    def measure(self, index: int, t: float, extra_states: Sequence = (), extra_values: Optional[Dict[str, Union[float, complex]]] = None) -> Dict[str, Union[float, complex]]:
        """Evaluate and record all observables.

        :param index: The results buffer index
        :type index: int
        :param t: The time (or step position) to record
        :type t: float
        :param extra_states: Additional states passed to observables.
        :type extra_states: tuple, optional
        :param extra_values: Additional label/value pairs to record (e.g. a DMRG sweep energy, which is not evaluated through a matrix element)
        :type extra_values: dict, optional
        :return:  Recorded values keyed by label.
        :rtype: dict[str, float or complex]
        """
        values = {obs.label: obs.evaluate(self.mel, self.A, *extra_states) for obs in self.observables}
        if extra_values:
            values.update(extra_values)
        self.results.record(index, t, values, maxchi=self.A.maximum_bond_dimension())
        return values

    def checkpoint(self, fname: str) -> None:
        """Save the current state to disk.

        :param fname: The output file name
        :type fname: str
        """
        self.A.save(fname)

    @abstractmethod
    def run(self):
        """Run this simulation to completion, returning its :class:`ResultsBuffer`."""

    def _flush(self, record_index: int) -> None:
        if self.output_file and record_index % self.output_stride == 0:
            self.results.to_hdf5(self.output_file)
        if self.checkpoint_file and self.checkpoint_stride and record_index % self.checkpoint_stride == 0:
            self.checkpoint(self.checkpoint_file)

    def _finalise(self) -> None:
        if self.output_file:
            self.results.to_hdf5(self.output_file)
        if self.checkpoint_file:
            self.checkpoint(self.checkpoint_file)

class TDVPSimulation(Simulation):
    """A :class:`Simulation` performing real- or imaginary-time dynamics with TDVP.

    :param A: The initial state
    :type A: ttn or ms_ttn
    :param H: The generator
    :type H: sop_operator or ms_sop_operator
    :param dt: The timestep used for integration
    :type dt: float
    :param nstep: The number of timesteps to perform
    :type nstep: int
    :param coefficient: Evolution coefficient (``-1j`` for real time,``-1`` for imaginary time)
    :type coefficient: complex, optional
    :param stride: Measurement interval, defaults to 1
    :type stride: int, optional
    :param output_file: Optional intermediate results file
    :type output_file: str, optional
    :param output_stride: Results flush interval, defaults to 10
    :type output_stride: int, optional
    :param checkpoint_file: Optional checkpoint file.
    :type checkpoint_file: str, optional
    :param checkpoint_stride: Checkpoint interval.
    :type checkpoint_stride: int, optional
    :param initial_ramp_steps: Number of logarithmic startup substeps.
    :type initial_ramp_steps: int, optional
    :param initial_ramp_scale: Relative size of the first startup substep., defaults to 1e-5
    :type initial_ramp_scale: float, optional
    """

    def __init__(self, A, H, dt: float, nstep: int, coefficient: Union[float, complex] = -1.0j, stride: int = 1, 
                integrator=None, expansion: str = "onesite", integrator_kwargs: Optional[dict] = None, mel=None, 
                observables: Optional[Sequence[Observable]] = None, reference_states: Optional[Sequence] = None, 
                system_modes=None, site_map: Optional[Dict[str, int]] = None, dtype=np.complex128, output_file: Optional[str] = None, 
                output_stride: int = 10, checkpoint_file: Optional[str] = None, checkpoint_stride: Optional[int] = None, 
                initial_ramp_steps: int = 5, initial_ramp_scale: float = 1e-5,
    ):
        super().__init__(A, H, integrator=integrator, expansion=expansion, integrator_kwargs=integrator_kwargs, mel=mel, 
                         observables=observables, reference_states=reference_states, system_modes=system_modes, site_map=site_map, 
                         nrecords=nstep // stride + 1, dtype=dtype,
        )
        self.integrator.dt = dt
        self.integrator.coefficient = coefficient

        self.dt = dt
        self.nstep = nstep
        self.stride = stride
        self.output_file = output_file
        self.output_stride = output_stride
        self.checkpoint_file = checkpoint_file
        self.checkpoint_stride = checkpoint_stride
        self.initial_ramp_steps = initial_ramp_steps
        self.initial_ramp_scale = initial_ramp_scale

    @staticmethod
    def _build_integrator(A, H, expansion: str, **kwargs):
        return tdvp(A, H, expansion=expansion, **kwargs)

    def run(self) -> ResultsBuffer:
        """Run the TDVP time evolution, measuring and recording observables as configured, and returning the populated :class:`ResultsBuffer`.

        :return: The results recorded throughout the run
        :rtype: ResultsBuffer
        """
        t = 0.0
        record_index = 0
        self.measure(record_index, t, extra_states=self.reference_states)
        record_index += 1

        for i in range(self.nstep):
            if i == 0 and self.initial_ramp_steps:
                t = _logarithmic_ramp(self.integrator, self.A, self.H, self.dt, t0=t, nstep=self.initial_ramp_steps, nscale=self.initial_ramp_scale)
                self.integrator.dt = self.dt
            else:
                self.integrator.step(self.A, self.H)
                t += self.dt

            if (i + 1) % self.stride == 0:
                self.measure(record_index, t, extra_states=self.reference_states)
                record_index += 1

                self._flush(record_index)

        self._finalise()
        return self.results


class DMRGSimulation(Simulation):
    """A :class:`Simulation` performing a ground state search with DMRG.

    :param A: The initial state
    :type A: ttn or ms_ttn
    :param H: The generator
    :type H: sop_operator or ms_sop_operator
    :param nsweep: The maximum number of sweeps to perform
    :type nsweep: int
    :param energy_tol: Optional energy convergence tolerance
    :type energy_tol: float, optional
    :param output_file: Optional intermediate results file
    :type output_file: str, optional
    :param output_stride: Results flush interval, defaults to 10
    :type output_stride: int, optional
    :param checkpoint_file: Optional checkpoint file.
    :type checkpoint_file: str, optional
    :param checkpoint_stride: Checkpoint interval.
    :type checkpoint_stride: int, optional
    """

    def __init__(self, A, H, nsweep: int, energy_tol: Optional[float] = None, integrator=None, expansion: str = "onesite", 
                 integrator_kwargs: Optional[dict] = None, mel=None, observables: Optional[Sequence[Observable]] = None, 
                 reference_states: Optional[Sequence] = None, system_modes=None, site_map: Optional[Dict[str, int]] = None, 
                 dtype=np.complex128, output_file: Optional[str] = None, output_stride: int = 10, checkpoint_file: Optional[str] = None, 
                 checkpoint_stride: Optional[int] = None,
    ):
        super().__init__(A, H, integrator=integrator, expansion=expansion, integrator_kwargs=integrator_kwargs, mel=mel, 
                         observables=observables, reference_states=reference_states, system_modes=system_modes, site_map=site_map, 
                         extra_labels=["E"], nrecords=nsweep, dtype=dtype,
        )

        self.nsweep = nsweep
        self.energy_tol = energy_tol
        self.output_file = output_file
        self.output_stride = output_stride
        self.checkpoint_file = checkpoint_file
        self.checkpoint_stride = checkpoint_stride

    @staticmethod
    def _build_integrator(A, H, expansion: str, **kwargs):
        return dmrg(A, H, expansion=expansion, **kwargs)

    def run(self) -> ResultsBuffer:
        """Run the DMRG ground state search, recording the energy (and any registered observables) at each sweep, stopping early if ``energy_tol`` is set and satisfied.

        :return: The results recorded throughout the run
        :rtype: ResultsBuffer
        """
        prev_energy = None
        record_index = 0

        for i in range(self.nsweep):
            self.integrator.step(self.A, self.H)
            energy = self.integrator.E()
            self.measure(record_index, i, extra_states=self.reference_states, extra_values={"E": energy})
            record_index += 1

            self._flush(record_index)

            if self.energy_tol is not None and prev_energy is not None and abs(energy - prev_energy) < self.energy_tol:
                prev_energy = energy
                break
            prev_energy = energy

        self._finalise()

        return self.results
