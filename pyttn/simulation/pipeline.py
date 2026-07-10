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

from typing import Callable, Dict, List, Optional, Tuple

from .results import ResultsBuffer
from .simulation import Simulation


class Pipeline:
    """Execute a sequence of simulation stages.

    Each stage receives the final state of the previous stage. Stages are added via :meth:`add_stage`; the first stage is called with ``state=None``.
    """

    def __init__(self):
        self._stages: List[Tuple[str, Callable[[object], Simulation], Optional[Callable[[object], object]]]] = []
        self.simulations: Dict[str, Simulation] = {}

    def add_stage(self, name: str, factory: Callable[[object], Simulation], transform: Optional[Callable[[object], object]] = None) -> "Pipeline":
        """Append a stage to this pipeline.

        :param name: Stage label.
        :type name: str
        :param factory: Callable that builds the stage's :class:`Simulation` from the previous stage's final state (or ``None`` for the first stage).
        :type factory: Callable[[ttn or ms_ttn or None], Simulation]
        :param transform: Optional transformation applied to the previous stage's final state before it is passed to ``factory``.
        :type transform: Callable[[ttn or ms_ttn], ttn or ms_ttn], optional
        :return: This pipeline
        :rtype: Pipeline
        """

        self._stages.append((name, factory, transform))
        return self

    def run(self):
        """Run all stages in sequence. Each stage receives the final state produced by the previous stage.

        :return: The final state produced by the last stage
        :rtype: ttn or ms_ttn
        """
        state = None
        for name, factory, transform in self._stages:
            if transform is not None and state is not None:
                state = transform(state)
            sim = factory(state)
            sim.run()
            self.simulations[name] = sim
            state = sim.state
        return state

    def results(self, name: str) -> ResultsBuffer:
        """Return the results recorded by a pipeline stage.

        :param name: The stage name
        :type name: str
        :return: The results buffer produced by running that stage
        :rtype: ResultsBuffer
        """
        return self.simulations[name].results
