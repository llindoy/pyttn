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

from opdictExt import OperatorDictionary
from ..operators.siteOperatorsExt import site_operator

class LabelledOperatorDictionary:
    """
    Lightweight wrapper providing label-based access to an OperatorDictionary.

    This class extends the standard mode-indexed operator dictionary by
    allowing operators to be accessed using string labels. Labels are mapped
    to one or more modes internally.

    The underlying OperatorDictionary is not modified.
    """

    def __init__(self, opdict: OperatorDictionary):
        """
        Initialise the labelled operator dictionary.

        :param opdict: The underlying operator dictionary
        :type opdict: OperatorDictionary
        """
        self._opdict = opdict
        self._labels: dict[str, set[int]] = {}

    def insert(self, mode: int, label: str, op: 'site_operator'):
        """
        Insert an operator for a given mode and label.

        :param mode: Mode index
        :type mode: int
        :param label: Label identifying the operator
        :type label: str
        :param op: The site_operator object
        :type op: site_operator
        """
        self._opdict.insert(mode, label, op)

        if label not in self._labels:
            self._labels[label] = set()
        self._labels[label].add(mode)

    def __getitem__(self, key):
        """
        Return an operator using label-based lookup.

        :param key: Either (mode, label) or label
        :type key: tuple[int, str] or str
        :returns: The corresponding site_operator
        :rtype: site_operator

        :raises KeyError: If the label is not found or is ambiguous
        :raises TypeError: If the key type is invalid
        """
        if isinstance(key, tuple):
            mode, label = key
            return self._opdict(mode, label)

        elif isinstance(key, str):
            modes = self._labels.get(key, set())

            if not modes:
                raise KeyError(f"Label '{key}' not found")

            if len(modes) > 1:
                raise KeyError(
                    f"Label '{key}' exists on multiple modes: {sorted(modes)}. "
                    "Use (mode, label)"
                )

            mode = next(iter(modes))
            return self._opdict(mode, key)

        else:
            raise TypeError("Key must be (mode, label) or label")

    def __setitem__(self, key, op):
        """
        Assign an operator using label-based syntax.

        :param key: Either (mode, label) or label
        :type key: tuple[int, str] or str
        :param op: The site_operator object
        :type op: site_operator

        :raises KeyError: If assignment by label is ambiguous
        :raises TypeError: If the key type is invalid
        """
        if isinstance(key, tuple):
            mode, label = key
            self.insert(mode, label, op)

        elif isinstance(key, str):
            modes = self._labels.get(key, set())

            if len(modes) != 1:
                raise KeyError(
                    f"Cannot assign '{key}' without unique mode"
                )

            mode = next(iter(modes))
            self._opdict.insert(mode, key, op)

        else:
            raise TypeError("Invalid key type")

    def nmodes(self):
        """
        Return the number of modes in the dictionary.

        :return: Number of modes
        :rtype: int
        """
        return self._opdict.nmodes()

    def __len__(self):
        """
        Return the number of modes in the dictionary.

        :return: Number of modes
        :rtype: int
        """
        return len(self._opdict)

    def backend(self):
        """
        Return the backend type of the dictionary.

        :return: The backend type
        :rtype: str
        """
        return self._opdict.backend()

    @property
    def dtype(self):
        """
        Return the dtype of the operator dictionary.

        :return: The dtype of the dictionary
        :rtype: numpy.dtype
        """
        return self._opdict.dtype

    def complex_dtype(self):
        """
        Return whether the dictionary stores complex-valued operators.

        :return: True if complex dtype, False otherwise
        :rtype: bool
        """
        return self._opdict.complex_dtype()

    def site_dictionary(self, mode: int):
        """
        Return the operator dictionary for a given mode.

        :param mode: Mode index
        :type mode: int
        :return: Mapping from labels to operators
        :rtype: dict[str, site_operator]
        """
        return self._opdict.site_dictionary(mode)


    def labels(self):
        """
        Return all labels present in the dictionary.

        :return: List of labels
        :rtype: list[str]
        """
        return list(self._labels.keys())

    def modes(self, label: str):
        """
        Return the modes associated with a label.

        :param label: Operator label
        :type label: str
        :return: List of mode indices
        :rtype: list[int]
        """
        return list(self._labels.get(label, []))

    def __str__(self):
        """
        Return the string representation of the dictionary.

        :return: String representation
        :rtype: str
        """
        return str(self._opdict)    