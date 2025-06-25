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

from abc import ABCMeta
from typing import Optional, Union
import numpy as np

from pyttn.ttnpp import matrix_element_complex, ttn_complex, ms_ttn_complex

try:
    from pyttn.ttnpp import matrix_element_real, ttn_real, ms_ttn_real

    _use_real_matel = True

except ImportError:
    _use_real_matel = False


class matrix_element(metaclass=ABCMeta):
    """A class defining the general interface for the pybind11 wrappers generated for the matrix_element object.  These wrapper classes are
    :class:`matrix_element_complex` and :class:`matrix_element_real` (with the real variant only present if the pybind11 wrapper has been built with support
    for real valued TTNs."""

    def __new__(
        cls,
        *args,
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = np.complex128,
        **kwargs,
    ) -> "matrix_element":
        """A factory method for constructing an object used for evaluating matrix elements from a TTN. If this function is passed a TTN object it uses
        this to construct a matrix_element suitable for evaluating matrix elements of this class.  Otherwise this will construct an empty matrix_element
        object with the required dtype.

        :param *args: A variable length list of arguments. Valid options are

            - none - in this case we call the default constructor of the matrix element class to construct an empty matrix element.
            - **A** (:class:`ttn` or :class:`ms_ttn`) - A TTN to define the topology and size of buffers needed to evaluate the tensor network.
            - **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Two TTNs with the same topology defining the size of the bra and ket respectively.
            - **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`), **H** (:class:`sop_operator` or :class:`ms_sop_operator`)- Two TTNs and a Hamiltonian operator with the same topology defining the size of the bra, ket and operator respectively
        
        :param dtype: The type to be stored in the matrix element object.  This is ignored if the a TTN object is passed in the first argument.
        :type dtype: {np.float64, np.complex128}, optional
        :param **kwargs: A dictionary containing optional input arguments.

            - **nbuffers** (int, optional) - The number of buffers to allocate.
            - **use_capacity** (bool, optional) - Whether or not to use the capacity of the TTNs to determine the size of the buffers.
        :returns: The matrix_element object
        :rtype: matrix_element_type
        """

        if _use_real_matel:
            if args:
                if isinstance(args[0], ttn_complex) or isinstance(args[0], ms_ttn_complex):
                    return matrix_element_complex(*args, **kwargs)
                elif isinstance(args[0], ttn_real) or isinstance(args[0], ms_ttn_real):
                    return matrix_element_real(*args, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype for matrix_element")
            else:
                if dtype == np.complex128 or dtype is complex:
                    return matrix_element_complex(**kwargs)
                elif dtype == np.float64 or dtype is float:
                    return matrix_element_real(**kwargs)
                else:
                    raise RuntimeError("Invalid dtype for matrix_element")

        else:
            if args:
                if isinstance(args[0], ttn_complex) or isinstance(args[0], ms_ttn_complex):
                    return matrix_element_complex(*args, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype for matrix_element")
            else:
                if dtype == np.complex128 or dtype is complex:
                    return matrix_element_complex(**kwargs)
                else:
                    raise RuntimeError("Invalid dtype for matrix_element")

    def assign(self, o):
        r"""Assign the value of this matrix element from another matrix element

        :param o: The other matrix_element object
        :type o: matrix_element
        """
        pass

    def clear(self):
        r"Clear and deallocate all internal buffers of the matrix element"
        pass

    def resize(self, *args, **kwargs):
        r"""Resize the matrix element to fit the required wavefunctions

        :param *args: A variable length list of arguments. Valid options are

            - **A** (:class:`ttn` or :class:`ms_ttn`) - A TTN to define the topology and size of buffers needed to evaluate the tensor network.
            - **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Two TTNs with the same topology defining the size of the bra and ket respectively.

        :type *args: [Arguments (variable number and type)]
        :param **kwargs: A dictionary containing optional input arguments.

            - **nbuffers** (int, optional) - The number of buffers to allocate.
            - **use_capacity** (bool, optional) - Whether or not to use the capacity of the TTNs to determine the size of the buffers.
        :type **kwargs: dict(Arguments (variable number and type))
        """
        pass

    def _call_(self, *args, use_sparsity=True):
        r"""Function for evaluating the inner product of tensor network states with matrix elements

        :param *args: A variable length list of arguments. See below for a list of valid arguments.
        :type *args: [Arguments (variable number and type)]
        :param use_sparsity: Whether or not to exploit sparsity in the evaluation of the matrix elements. (Default: True)
        :type use_sparsity: bool, optional

        :*args options:
        Valid options for the argument list depend on the type of quantity we are evaluating.

        :math:`\langle A | A \rangle`:

            - **A** (:class:`ttn` or :class:`ms_ttn`) - Compute the inner product of the ttn with itself.

        :math:`\langle A | O | A \rangle`:

            - **op** (:class:`site_operator`), **A** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the single site operator op.
            - **op** (:class:`site_operator`), **mode** (int), **A** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the single site operator op acting on mode mode.
            - **op** (list[:class:`site_operator`]), **A** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of site operator op.
            - **op** (list[:class:`site_operator`]), **modes** (list[int]), **A** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of site operator op.  Each site operator acts on the corresponding mode stored in the modes list.
            - **op** (:class:`product_operator`), **A** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of site operator op.
            - **op** (:class:`ms_sop_operator` or :class:`ms_sop_operator`), **A** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of sum-of-product operator op.

        :math:`\langle A | B \rangle`:

            - **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Compute the overlad .

        :math:`\langle A | O | B \rangle`:

            - **op** (:class:`site_operator`), **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Compute the matrix element of the single site operator op.
            - **op** (:class:`site_operator`), **mode** (int), **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the single site operator op acting on mode mode.
            - **op** (list[:class:`site_operator`]), **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of site operator op.
            - **op** (list[:class:`site_operator`]), **modes** (list[int]), **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of site operator op.  Each site operator acts on the corresponding mode stored in the modes list.
            - **op** (:class:`product_operator`), **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of site operator op.
            - **op** (:class:`ms_sop_operator` or :class:`ms_sop_operator`), **A** (:class:`ttn` or :class:`ms_ttn`), **B** (:class:`ttn` or :class:`ms_ttn`) - Compute the expectation value of the product of sum-of-product operator op.
        """
        pass

matrix_element.register(matrix_element_complex)
if _use_real_matel:
    matrix_element.register(matrix_element_real)
