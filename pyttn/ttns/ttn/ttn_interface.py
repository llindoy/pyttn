# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np


class ttn_dtype:
    r"""A class defining the general interface for the pybind11 wrappers generated for the ttn object.  These wrapper classes are
    :class:`ttn_complex` and :class:`ttn_real` (with the real variant only present if the pybind11 wrapper has been built with support
    for real valued TTNs.

    :param *args: A variable length list of arguments. Valid options are

        - none - in this case we call the default constructor of the ttn class to construct an empty ttn.
        - **A** (:class:`ms_ttn_slice_complex` or :class:`ms_ttn_slice_real`) - Construct a ttn to represent a single slice of a multiset TTN
        - **A** (:class:`ttn_complex` or :class:`ttn_real`) - Construct the ttn object from another ttn object potentially converting the dtype
        - **topology** (:class:`ntree` or str) - Construct a ttn from a ntree object defining the topology and bond dimensions of the ttn
        - **topology** (:class:`ntree` or str), **capacity** (:class:`ntree` or str ) - Construct a ttn from an ntree object defining the topology and a capacity defining the maximum bond dimensions

    :type *args: [Arguments (variable number and type)]
    :param **kwargs: A dictionary containing optional input arguments.

        - **purification** (bool) - Whether or not this state should represent a purification of a state.
    :type **kwargs: dict(Arguments (variable number and type))
    """

    def __init__(self, *args, dtype=np.complex128, **kwargs):
        raise RuntimeError(
            "The ttn_dtype class is not constructable.  This class is present to provide cleaner documentation for the pybind11 classes.")

    @property
    def complex_dtype(self):
        r"""Return the data type stored in the ttn

        :returns: dtype
        :rtype: {np.complex128 or np.float64}
        """
        pass

    def assign(self, o):
        r"""Assign the value of this ttn from another ttn

        :param o: The other ttn object
        :type o: ttn_dtype or ttn_real
        """
        pass

    def bond(self):
        r"""Return a list of all bonds in the network

        :returns: All bonds in the network
        :rtype: list[int, int]
        """
        pass

    def bond_dimensions(self):
        r"""Return a dictionary containing the bond (the two sites forming the bond) and bond dimension of all bonds in the network

        :returns: All bond dimensions in the network
        :rtype: dict([int, int], int)
        """
        pass

    def bond_capacities(self):
        r"""Return a dictionary containing the bond (the two sites forming the bond) and maximum bond dimension of all bonds in the network

        :returns: All maximum bond dimensions in the network
        :rtype: dict([int, int], int)
        """
        pass


    def reset_orthogonality_centre(self):
        r"""Resets the orthogonality centre of the TTN to the root node of the tree."""
        pass

    def resize(self, *args, purification=False):
        r"""Resize the TTN object given a new set of topology information. This optionally takes a flag allowing for the state to automatically represent a purification of a wavefunction

        :param *args: A variable length list of arguments. Valid options are

            - **topology** (:class:`ntree` or str) - Construct a ttn from a ntree object defining the topology and bond dimensions of the ttn
            - **topology** (:class:`ntree` or str), **capacity** (ntree or str ) - Construct a ttn from an ntree object defining the topology and a capacity defining the maximum bond dimensions
        :type *args: [Arguments (variable number and type)]
        :param purification: Whether or not the buffers should be resized to store a purification of the requested state size.  (Default: False)
        :type purification: bool, optional
        """
        pass

    def set_seed(self, seed):
        r"""Set the value of the random number generate seed used for internal operations requiring random sampling

        :param seed: The new value of the seed
        :type seed: int
        """
        pass

    def set_state(self, state, random_unoccupied_initialisation=False):
        r"""Set the coefficients in the TTN so that it represents a user specified product state 

        :param state: The occupation number state to set the TTN to
        :type state: list[int]
        :param random_unoccupied_initialisation: Whether or not to set all other elements of the TTN not determining the product state to random values or not. (Default: False)
        :type random_unoccupied_initialisation: bool, optional
        """
        pass

    def set_product(self, state):
        r"""Set the coefficients in the TTN so that it represents a product of a set of one body states 

        :param state: A list containing a set of vectors corresponding to the individual product states
        :type state: list[list[dtype]]
        """
        pass

    def set_identity_purification(self):
        r"""Sets the state of the TTN to a purification state representing the identity
        """
        pass

    def sample_product(self, dist):
        r"""Sample a direct product of occupation states from a set of probabilities of observing each mode in a given state 

        :param state: A list containing a set of vectors corresponding to the probabilities of observing each occupation state
        :type state: list[list[dtype]]
        """
        pass

    def __imul__(self, b):
        r"""Inplace multiplication of the TTN object by a scalar

        :param b: Scalar value to multiply TTN by
        :type b: number
        """
        pass

    def __idiv__(self, b):
        r"""Inplace division of the TTN object by a scalar

        :param b: Scalar value to divide TTN by
        :type b: number
        """
        pass

    def conj(self):
        "Take the complex conjugate of the TTN.  Here this is evaluated lazily"
        pass

    def random(self):
        "Sample the coefficients in the TTN randomly from a normal distribution"
        pass

    def zero(self):
        "Set all coefficients in the TTN to zero"
        pass

    def clear(self):
        "Clear and deallocate all internal buffers of the TTN"
        pass

    def __iter__(self):
        r"""
        :returns: Iterator object over nodes in TTN
        :rtype: iterator
        """
        pass

    def mode_dimensions(self):
        r"""
        :returns: List of local Hilbert space dimensions
        :rtype: list[int]
        """
        pass

    def dim(self, i):
        r"""Returns the local Hilbert space dimension of mode i

        :param i: The index of the mode
        :type i: int

        :returns: local Hilbert space dimension of mode i
        :rtype: int
        """
        pass

    def nmodes(self):
        r"""
        :returns: The number of modes in the TTN
        :rtype: int
        """
        pass

    def is_purification(self):
        r"""
        :returns: Whether or not the state represents a purification
        :rtype: bool
        """
        pass

    def ntensors(self):
        r"""
        :returns: The total number of tensors in the tensor network
        :rtype: int
        """
        pass

    def nsites(self):
        r"""
        :returns: The total number of tensors in the tensor network
        :rtype: int
        """
        pass

    def nset(self):
        r"""
        :returns: The number of set variables for the TTN.  Here it is one
        :rtype: int
        """
        pass

    def nelem(self):
        r"""
        :returns: The total number of elements in all tensors of the network.
        :rtype: int
        """
        pass

    def __len__(self):
        r"""
        :returns: The number of modes in the TTN
        :rtype: int
        """
        pass

    def compute_maximum_bond_entropy(self):
        r"""Computes the maximum SvN across any bond in the tensor network and returns the results

        :returns: The maximum bond entropy in the tensor network
        :rtype: float

        """
        pass

    def maximum_bond_entropy(self):
        r"""Returns the previously computed maximum SvN across any bond in the tensor network

        :returns: The maximum bond entropy in the tensor network
        :rtype: float

        """
        pass

    def bond_entropy(self, i):
        r"""Returns the SvN across the ith bond of the current orthogonality centre. 
        Where for all nodes but the root 0 corresponds to the parent of the current orthogonality centre and its children are then 1-nchild,
        For the root i just indexes the children

        :returns: The bond entropy 
        :rtype: float

        """
        pass

    def maximum_bond_dimension(self):
        r"""            
        :returns: The maximum bond dimension
        :rtype: int

        """
        pass

    def minimum_bond_dimension(self):
        r"""            
        :returns: The minimum bond dimension
        :rtype: int

        """
        pass

    def has_orthogonality_centre(self):
        r"""            
        :returns: Whether or not the TTN has an active orthogonality centre
        :rtype: bool

        """
        pass

    def orthogonality_centre(self):
        r"""            
        :returns: The index of the current orthogonality centre
        :rtype: int

        """
        pass

    def is_orthogonalised(self):
        r"""            
        :returns: Whether or not the TTN has an orthogonality centre at the root
        :rtype: bool

        """
        pass

    def force_set_orthogonality_centre(self, i):
        r"""Sets the orthogonality centre of the tensor network to index i but does not modify the tensor to ensure that this is a
        valid orthogonality centre

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i: int or list[int]

        """
        pass

    def shift_orthogonality_centre(self, i, tol=0, nchi=0):
        r"""Shift the orthogonality centre down the ith bond of the current orthogonality centre with possible truncation. 
        Where for all nodes but the root 0 corresponds to the parent of the current orthogonality centre and its children are then 1-nchild
        For the root i just indexes the children

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i: int or list[int]
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    def set_orthogonality_centre(self, i, tol=0, nchi=0):
        r"""Sets the orthogonality centre of the tensor network to index i either introducing an orthogonality centre if there is none
        or simply shifting the orthogonality centre from its current location to the required location

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i: int or list[int]
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    def orthogonalise(self, force=False):
        r"""Shifts the orthogonality centre to the root node of the TTN

        :param force: Whether or not to force a full reorthogonalisation of the TTN regardless of whether or not it believes it has an orthogonality centre
        :type force: bool, optional

        """

    def truncate(self, tol=0, nchi=0):
        r"""Ensures the tensor network is in an orthogonalised form.  Then performs an euler tour truncating each bond according to the user
        specified tol and nchi parameters

        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    def normalise(self):
        r"""Ensures the TTN is a normalised to one and returns the previous value of the norm of the tensor

        :returns: The previous 2-norm of the TTN
        :rtype: float
        """
        pass

    def norm(self):
        r"""
        :returns: The 2-norm of the TTN
        :rtype: float
        """
        pass

    def __setitem__(self, i, v):
        r"""Sets the value of a site tensor in the tensor network

        :param i: Index of the node to set
        :type i: int
        :param v: The new value of the node data object
        :type v: ttn_data_dtype

        """
        pass

    def __getitem__(self, i, v):
        r"""Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :returns: tensor data
        :rtype: ttn_data_dtype

        """
        pass

    def set_site_tensor(self, i, v):
        r"""Sets the value of a site tensor in the tensor network

        :param i: Index of the node to set
        :type i: int
        :param v: The new value of the node data object
        :type v: ttn_data_dtype

        """
        pass

    def site_tensor(self, i, v):
        r"""Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :returns: tensor data
        :rtype: np.ndarray or linalg.matrix

        """
        pass

    def measure_without_collapse(self, i):
        r"""Evaluate the probablity of observing each state following a projective measurement applied to mode i without performing the collapse

        :param i: The physical mode to perform the projective measurement on
        :type i: int

        :returns: The probability of observing each basis state following the projective measurement
        :rtype: list[float]
        """
        pass

    def collapse_basis(self, U, truncate=True, tol=0, nchi=0):
        r"""Perform a projective measurement across all modes in the TTN applying a basis transformation U_i to each mode i before doing so

        :param U: A list of basis transformations to apply to the state before performing the projective measurement
        :type U: list[np.ndarray] or list[linalg.matrix]
        :param truncate: Whether or not to truncate the state following collapse as it is a product state. (Default: True)
        :type truncate: bool, optional
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        :returns: The probability of this collapse event occurint
        :rtype: float
        """
        pass

    def collapse(self, truncate=True, tol=0, nchi=0):
        r"""Perform a projective measurement across all modes in the TTN 

        :param truncate: Whether or not to truncate the state following collapse as it is a product state. (Default: True)
        :type truncate: bool, optional
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        :returns: The probability of this collapse event occurint
        :rtype: float
        """
        pass

    def apply_one_body_operator(self, *args, shift_orthogonality=True):
        r"""Apply a one-body operator to the TTN updating its value

        :param *args: A variable length list of arguments. Valid options are

            - **op** (linalg.matrix or np.ndarray or site_operator_dtype), **mode** (int) -  Apply the operator op to mode mode
            - **op** (site_operator_dtype) - Apply the operator op to the mode specified by op
        :type *args: [Arguments (variable number and type)]
        :param shift_orthogonality: Whether or not to shift the orthogonality centre of the TTN to the leaf node that will be updated by this one-body operator.  (Default: True)
        :type shift_orthogonality: bool, optional
        """
        pass

    def apply_product_operator(self, op, shift_orthogonality=True):
        r"""Apply a product of one-body operator to the TTN updating its value

        :param op: The product operator to apply to the system
        :type op: product_operator_dtype
        :param shift_orthogonality: Whether or not to shift the orthogonality centre of the TTN to the leaf node that will be updated by this one-body operator.  (Default: True)
        :type shift_orthogonality: bool, optional
        """
        pass

    def apply_operator(self, op, shift_orthogonality=True):
        r"""Apply a product of one-body operator to the TTN updating its value

        :param op: The product operator to apply to the system
        :type op: site_operator_dtype or product_operator_dtype
        :param shift_orthogonality: Whether or not to shift the orthogonality centre of the TTN to the leaf node that will be updated by this one-body operator.  (Default: True)
        :type shift_orthogonality: bool, optional
        """
        pass

    def __imatmul__(self, op):
        r"""Apply an operator to the TTN updating its value.  Shifting the orthogonality centre to the leaf nodes that will be updated by this operator

        :param op: The product operator to apply to the system
        :type op: site_operator_dtype or product_operator_dtype or sop_opertor_dtype

        """
        pass

    def __rmatmul__(A,  op):
        r"""Apply an operator to the TTN updating its value, returning the result as a new TTN

        :param A: The ttn to apply the operator to
        :type A: ttn_dtype
        :param op: The product operator to apply to the system
        :type op: site_operator_dtype or product_operator_dtype or sop_opertor_dtype

        :returns: The result of op@A
        :rtype: ttn_dtype

        """
        pass
