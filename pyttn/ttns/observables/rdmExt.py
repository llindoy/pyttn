# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np

from pyttn.ttnpp import rdm_complex, ms_ttn_complex, ttn_complex

try:
    from pyttn.ttnpp import rdm_real, ms_ttn_real, ttn_real

    _use_real_matel = True

except ImportError:
    _use_real_matel = False


class RDM(metaclass=ABCMeta):
    """A class defining the general interface for the pybind11 wrappers generated for the RDM object.  These wrapper classes are
    :class:`rdm_complex` and :class:`rdm_real` (with the real variant only present if the pybind11 wrapper has been built with support
    for real valued TTNs."""

    def __new__(
        cls,
        *args,
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = np.complex128,
        **kwargs,
    ) -> "RDM":
        """A factory method for constructing an object used for evaluating rdm objects from a TTN. If this function is passed a TTN object it uses
        this to construct a RDM suitable for evaluating rdm objects of this class.  Otherwise this will construct an empty RDM
        object with the required dtype.

        :param A: A variable length list of arguments. Valid options are

            - none - in this case we call the default constructor of the rdm class to construct an empty rdm object.
            - **A** (:class:`ttn` ) - A TTN to define the topology and size of buffers needed to evaluate the tensor network.bra,
        
        :param dtype: The type to be stored in the rdm object object.  This is ignored if the a TTN object is passed in the first argument.
        :type dtype: {np.float64, np.complex128}, optional
        :param `**kwargs`: A dictionary containing optional input arguments.
            - **two_body** (bool, optional) - Whether or not the buffers for computing two-body RDMs should be allocated
        :returns: The RDM object
        :rtype: RDM
        """

        if _use_real_matel:
            if args:
                if isinstance(args[0], ttn_complex) :
                    return rdm_complex(*args, **kwargs)
                elif isinstance(args[0], ttn_real):
                    return rdm_real(*args, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype for RDM")
            else:
                if dtype == np.complex128 or dtype is complex:
                    return rdm_complex(**kwargs)
                elif dtype == np.float64 or dtype is float:
                    return rdm_real(**kwargs)
                else:
                    raise RuntimeError("Invalid dtype for RDM")

        else:
            if args:
                if isinstance(args[0], ttn_complex):
                    return rdm_complex(*args, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype for RDM")
            else:
                if dtype == np.complex128 or dtype is complex:
                    return rdm_complex(**kwargs)
                else:
                    raise RuntimeError("Invalid dtype for RDM")
                
    @abstractmethod
    def assign(self, o):
        """Assign the value of this rdm object from another rdm object

        :param o: The other RDM object
        :type o: RDM
        """
        pass

    @abstractmethod
    def clear(self):
        "Clear and deallocate all internal buffers of the rdm object"
        pass

    @abstractmethod
    def resize(self, *args, **kwargs):
        """Resize the rdm object to fit the required wavefunctions

        :param `*args`: A variable length list of arguments. Valid options are

            - **A** (:class:`ttn` or :class:`ms_ttn`) - A TTN to define the topology and size of buffers needed to evaluate the tensor network.
            - **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Two TTNs with the same topology defining the size of the bra and ket respectively.

        :param `**kwargs`: A dictionary containing optional input arguments.

            - **nbuffers** (int, optional) - The number of buffers to allocate.
            - **use_capacity** (bool, optional) - Whether or not to use the capacity of the TTNs to determine the size of the buffers.
        """
        pass

    @abstractmethod
    def _call_(self, A, *args, use_sparsity=True):
        """Function for evaluating the inner product of tensor network states with rdm objects

        :param `*args`: A variable length list of arguments. See below for a list of valid arguments.


        """
        pass

RDM.register(rdm_complex)
if _use_real_matel:
    RDM.register(rdm_real)

rdm = RDM