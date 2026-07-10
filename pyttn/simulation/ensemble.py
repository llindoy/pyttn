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

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .simulation import Simulation


def _run_sample(sample_fn: Callable[[int], Simulation], index: int) -> Dict[str, np.ndarray]:
    """Build and run one sample's :class:`Simulation`, returning only its results.

    This is a module-level function (rather than an :class:`Ensemble` method) so that
    it can be sent to a worker process by :class:`concurrent.futures.ProcessPoolExecutor`
    without needing to pickle the :class:`Simulation`, state, or integrator objects
    themselves - the state is built fresh inside the worker, and only the plain numpy
    result arrays cross back over the process boundary.
    """
    sim = sample_fn(index)
    sim.run()
    return sim.results.as_dict()


class Ensemble:
    """Runs many independent :class:`Simulation` samples, optionally in parallel,
    for algorithms that require sampling over multiple trajectories (e.g. METTS-style
    thermal sampling).

    Each sample is built entirely independently by ``sample_fn`` - its own state, its
    own randomness derived from the sample index - mirroring how sampling algorithms
    such as METTS actually work (every sample restores from the same checkpoint and
    applies fresh randomness, rather than continuing a running trajectory).

    :param sample_fn: Builds the :class:`Simulation` for sample ``index``; called
        with a plain ``int`` in ``range(n_samples)``
    :type sample_fn: Callable[[int], Simulation]
    :param n_samples: The number of samples to run
    :type n_samples: int
    :param n_workers: The number of worker processes to run samples in; 1 (default)
        runs samples serially in the calling process
    :type n_workers: int, optional
    :param aggregate: If given, applied to the list of per-sample results (each a
        dict of numpy arrays, as returned by :meth:`ResultsBuffer.as_dict`) to
        produce a single combined result, e.g. a sample average
    :type aggregate: Callable[[list[dict[str, np.ndarray]]], Any], optional
    """

    def __init__(self, sample_fn: Callable[[int], Simulation], n_samples: int, n_workers: int = 1, aggregate: Optional[Callable[[List[Dict[str, np.ndarray]]], Any]] = None):
        self.sample_fn = sample_fn
        self.n_samples = n_samples
        self.n_workers = n_workers
        self.aggregate = aggregate

    def run(self) -> Any:
        """Run every sample, returning either the raw list of per-sample results, or
        the output of ``aggregate`` applied to that list if one was given.

        :return: The per-sample results, or their aggregate
        :rtype: list[dict[str, np.ndarray]] or Any
        """
        if self.n_workers <= 1:
            results = [_run_sample(self.sample_fn, i) for i in range(self.n_samples)]
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(_run_sample, repeat(self.sample_fn), range(self.n_samples)))

        if self.aggregate is not None:
            return self.aggregate(results)
        return results
