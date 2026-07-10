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

from typing import Optional, Union


class Observable:
    """A labelled quantity to be measured during a :class:`Simulation`.

    Associates an operator (or ``None`` for a norm/overlap) with a label used when 
    recording results. Evaluation is performed by :meth:`evaluate` using the states
    supplied by the caller.

    :param label: Observable label
    :type label: str
    :param op: Operator to evaluate, or ``None`` to measure a norm/overlap.
    :type op: site_operator, list[site_operator], product_operator, sop_operator, ms_sop_operator, optional
    :param mode: Mode acted on by ``op`` when ``op`` is a single  :class:`site_operator`.
    :type mode: int, optional
    """

    def __init__(self, label: str, op=None, mode: Optional[int] = None):
        self.label = label
        self.op = op
        self.mode = mode

    def evaluate(self, mel, *states) -> Union[float, complex]:
        """Evaluate this observable for the given state(s).

        :param mel: The matrix element evaluator
        :type mel: pyttn.matrix_element
        :param `*states`: State(s) used to evaluate the observable.
        :type states: ttn or ms_ttn
        :return: The evaluated matrix element
        :rtype: float or complex
        """
        if not states:
            raise ValueError(f"Observable '{self.label}' requires at least one state to evaluate.")

        if self.op is None:
            return mel(*states)
        if self.mode is None:
            return mel(self.op, *states)
        return mel(self.op, self.mode, *states)

    def __repr__(self) -> str:
        return f"Observable(label={self.label!r}, op={self.op!r}, mode={self.mode!r})"
