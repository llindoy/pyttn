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

from typing import Dict, List, Optional, Union

import numpy as np


class ResultsBuffer:
    """Pre-allocated storage for simulation results.

    Stores a ``t`` column, a ``maxchi`` column, and one column per observable. Results can be exported to HDF5.

    :param labels: Observable labels.
    :type labels: list[str]
    :param nrecords: Maximum number of records.
    :type nrecords: int
    :param dtype: Observable data type.
    :type dtype: type, optional
    """

    def __init__(self, labels: List[str], nrecords: int, dtype=np.complex128):
        self.nrecords = nrecords
        self.dtype = dtype
        self.count = 0
        self.t = np.zeros(nrecords)
        self.maxchi = np.zeros(nrecords)
        self.data: Dict[str, np.ndarray] = {label: np.zeros(nrecords, dtype=dtype) for label in labels}

    def record(self, index: int, t: float, values: Dict[str, Union[float, complex]], maxchi: Optional[float] = None) -> None:
        """Record measurements at a given index.
         
        :param index: Record index.
        :type index: int
        :param t: Time or step coordinate.
        :type t: float
        :param values: Observable values keyed by label.
        :type values: dict[str, float | complex]
        :param maxchi: Maximum bond dimension.
        :type maxchi: float, optional       
        """
        if index >= self.nrecords:
            raise IndexError(f"Record index {index} out of range for buffer of size {self.nrecords}.")

        self.t[index] = t
        if maxchi is not None:
            self.maxchi[index] = maxchi
        for label, value in values.items():
            if label not in self.data:
                raise KeyError(f"Unknown observable label '{label}'; expected one of {list(self.data.keys())}.")
            self.data[label][index] = value
        self.count = max(self.count, index + 1)

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Return all recorded data as a dictionary of numpy arrays.

        :return: Recorded columns keyed by name.
        :rtype: dict[str, np.ndarray]
        """
        result = {"t": self.t[: self.count], "maxchi": self.maxchi[: self.count]}
        result.update({label: arr[: self.count] for label, arr in self.data.items()})
        return result

    def to_hdf5(self, fname: str, mode: str = "w", attrs: Optional[Dict[str, object]] = None) -> None:
        """Write recorded data to an HDF5 file.

        :param fname: Output filename.
        :type fname: str
        :param mode: HDF5 file mode., defaults to "w" (overwrite)
        :type mode: str, optional
        :param attrs: Optional file attributes.
        :type attrs: dict, optional
        """
        import h5py

        with h5py.File(fname, mode) as h5:
            for name, data in self.as_dict().items():
                h5.create_dataset(name, data=data)
            if attrs:
                for key, value in attrs.items():
                    h5.attrs[key] = value
