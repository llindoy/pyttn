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

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np

from pyttn.ttnpp import ms_ttn_complex

# and attempt to import the real ttns
try:
    from pyttn.ttnpp import ms_ttn_real

    _real_ttn_import = True
except ImportError:
    _real_ttn_import = False

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import ttn_complex as ms_ttn_complex_cuda

    _cuda_import = True
    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import ttn_real as ms_ttn_real_cuda

except ImportError:
    _cuda_import = False


def is_ms_ttn(A) -> bool:
    """A function for determining whether a given object is a multiset ttn
    :param A: The object to test
    :returns: Whether or not the object is a ms_ttn
    :rtype: bool
    """
    ret = isinstance(A, ms_ttn_complex)
    if _real_ttn_import:
        ret = ret or isinstance(A, ms_ttn_real)

    if _cuda_import:
        ret = ret or isinstance(A, ms_ttn_complex_cuda)

        if _real_ttn_import:
            ret = ret or isinstance(A, ms_ttn_real_cuda)

    return ret


def is_multiset_ttn(A) -> bool:
    """A function for determining whether a given object is a multiset ttn
    :param A: The object to test
    :returns: Whether or not the object is a ms_ttn
    :rtype: bool
    """
    return is_ms_ttn(A)


def _ms_ttn_blas(*args, dtype=np.complex128, **kwargs):
    if args:
        if isinstance(args[0], ms_ttn_complex):
            return ms_ttn_complex(*args, **kwargs)
        elif _real_ttn_import and isinstance(args[0], ms_ttn_real):
            if dtype == np.complex128 or dtype is complex:
                return ms_ttn_complex(*args, **kwargs)
            else:
                return ms_ttn_real(*args, **kwargs)
        else:
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                return ms_ttn_complex(*args, **kwargs)
            elif dtype == np.float64 or dtype is float:
                return ms_ttn_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ms_ttn")
    else:
        if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
            return ms_ttn_complex(*args, **kwargs)
        elif dtype == np.float64 or dtype is float:
            return ms_ttn_real(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ms_ttn")


def _ms_ttn_cuda(*args, dtype=np.complex128, **kwargs):
    if args:
        if isinstance(args[0], ms_ttn_complex_cuda):
            return ms_ttn_complex_cuda(*args, **kwargs)
        elif _real_ttn_import and isinstance(args[0], ms_ttn_real):
            if dtype == np.complex128 or dtype is complex:
                return ms_ttn_complex_cuda(*args, **kwargs)
            else:
                return ms_ttn_real_cuda(*args, **kwargs)
        else:
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                return ms_ttn_complex_cuda(*args, **kwargs)
            elif dtype == np.float64 or dtype is float:
                return ms_ttn_real_cuda(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ms_ttn")
    else:
        if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
            return ms_ttn_complex_cuda(*args, **kwargs)
        elif dtype == np.float64 or dtype is float:
            return ms_ttn_real_cuda(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ms_ttn")





class multiset_ttn(metaclass=ABCMeta):
    """Class for handling multiset tree tensor network state operator
    """   
    def __new__(cls, *args,
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = np.complex128,
        backend: str = "blas",
        **kwargs,
    ) -> 'multiset_ttn':
        r"""Factory function for constructing a multiset tree tensor network state operator

        :param *args: Variable length list of arguments. This function can handle two possible lists of arguments

            - Default construct a multiset TTN object
            - msttn (:class:`multiset_ttn`) - Copy construct a multiset TTN object
            - tree (:class:`ntree`), nset (int) - Construct a multiset TTN from an Ntree object and the number of set variables
            - string (str), nset (int) - Construct multiset TTN from an string defining an Ntree object and the number of set variables

        :param dtype: The dtype to use for the ttn.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the ttn.  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :param **kwargs: Additional keyword arguments that are based to the ttn object constructor
        :returns: The Multiset Tree Tensor Network State object
        :rtype: multiset_ttn 
        """   
        if backend == "blas":
            return _ms_ttn_blas(*args, dtype=dtype, **kwargs)
        elif _cuda_import and backend == "cuda":
            return _ms_ttn_cuda(*args, dtype=dtype, **kwargs)
        else:
            raise RuntimeError("Invalid backend type for multiset_ttn")


    @abstractmethod
    def complex_dtype(self):
        """Return the data type stored in the multiset ttn

        :returns: dtype
        :rtype: {np.complex128 or np.float64}
        """
        pass

    @abstractmethod
    def nthreads(self):
        """Stores the number of threads that can be used to attempt to parallelise updates over the set variables

        :returns: dtype
        :rtype: {np.complex128 or np.float64}
        """
        pass

    @abstractmethod    
    def assign(self, o):
        """Assign the value of this multiset ttn from another multiset ttn

        :param o: The other multiset ttn object
        :type o: multiset_ttn
        """
        pass

    @abstractmethod
    def slice(self, i):
        """Returns a slice object that allows for easy accessing of the multiset ttn correspond to a single set variable i

        :param i: The set index of the slice
        :type i: int

        :return: A slice of the multiset ttn used for accessing the tensor network associated with a single system state
        :rtype: multiset_ttn_slice

        """

    @abstractmethod
    def bond(self):
        """Return a list of all bonds in the network

        :returns: All bonds in the network
        :rtype: list[int, int]
        """
        pass

    @abstractmethod
    def bond_dimensions(self):
        """Return a dictionary containing the bond (the two sites forming the bond) and bond dimension of all bonds in the network

        :returns: All bond dimensions in the network
        :rtype: dict([int, int], list[int])
        """
        pass

    @abstractmethod
    def bond_capacities(self):
        """Return a dictionary containing the bond (the two sites forming the bond) and maximum bond dimension of all bonds in the network

        :returns: All maximum bond dimensions in the network
        :rtype: dict([int, int], list[int])
        """
        pass

    @abstractmethod
    def reset_orthogonality_centre(self):
        """Resets the orthogonality centre of the multiset TTN to the root node of the tree."""
        pass

    @abstractmethod
    def resize(self, *args, nset, purification=False):
        """Resize the multiset TTN object given a new set of topology information. This optionally takes a flag allowing for the state to automatically represent a purification of a wavefunction

        :param *args: A variable length list of arguments. Valid options are

            - **topology** (:class:`ntree` or str) - Construct a multiset ttn from a ntree object defining the topology and bond dimensions of the ttn
            - **topology** (:class:`ntree` or str), **capacity** (ntree or str ) - Construct a multiset ttn from an ntree object defining the topology and a capacity defining the maximum bond dimensions
        :type *args: [Arguments (variable number and type)]
        :param nset: The number of set variables to use for the multiset_ttn
        :type nset: int
        :param purification: Whether or not the buffers should be resized to store a purification of the requested state size.  (Default: False)
        :type purification: bool, optional
        """
        pass

    @abstractmethod
    def set_seed(self, seed):
        """Set the value of the random number generate seed used for internal operations requiring random sampling

        :param seed: The new value of the seed
        :type seed: int
        """
        pass

    @abstractmethod
    def set_state(self, *args, random_unoccupied_initialisation=False):
        """Set the coefficients in the multiset TTN so that it represents a user specified product state

        :param *args: A variable length list of arguments. Valid options are

            - **set_index** (int), **state** (list[int]) - For setting the state to be a product state vector acting on a specific set index
            - **coeff** (list[dtype]), **state** (list[list[int]]) - For setting the system to be the state :math:`\sum_i c_i \|i\rangle \bigotimes \mathrm{state}_i`

        :param random_unoccupied_initialisation: Whether or not to set all other elements of the multiset TTN not determining the product state to random values or not. (Default: False)
        :type random_unoccupied_initialisation: bool, optional
        """
        pass

    # def set_product(self, state):
    #    """Set the coefficients in the multiset TTN so that it represents a product of a set of one body states

    #    :param *args: A variable length list of arguments. Valid options are
    #
    #        - **set_index** (int), **state** (list[list[dtype]]) - For setting the state to be a product state vector acting on a specific set index
    #        - **coeff** (list[dtype]), **state** (list[list[list[dtype]]]) - For setting the system to be the state :math:`\sum_i c_i \|i\rangle \bigotimes \mathrm{state}_i`

    #    """
    #    pass

    # def set_identity_purification(self):
    #    """Sets the state of the multiset TTN to a purification state representing the identity
    #    """
    #    pass

    # def sample_product(self, dist):
    #    """Sample a direct product of occupation states from a set of probabilities of observing each mode in a given state

    #    :param state: A list containing a set of vectors corresponding to the probabilities of observing each occupation state
    #    :type state: list[list[dtype]]
    #    """
    #    pass

    @abstractmethod
    def __imul__(self, b):
        """Inplace multiplication of the multiset TTN object by a scalar

        :param b: Scalar value to multiply multiset TTN by
        :type b: number
        """
        pass

    @abstractmethod
    def __idiv__(self, b):
        """Inplace division of the multiset TTN object by a scalar

        :param b: Scalar value to divide multiset TTN by
        :type b: number
        """
        pass

    @abstractmethod
    def conj(self):
        "Take the complex conjugate of the multiset TTN.  Here this is evaluated lazily"
        pass

    @abstractmethod
    def random(self):
        "Sample the coefficients in the multiset TTN randomly from a normal distribution"
        pass

    @abstractmethod
    def zero(self):
        "Set all coefficients in the multiset TTN to zero"
        pass

    @abstractmethod    
    def clear(self):
        "Clear and deallocate all internal buffers of the multiset TTN"
        pass

    @abstractmethod
    def __iter__(self):
        """
        :returns: Iterator object over nodes in multiset TTN
        :rtype: iterator
        """
        pass

    @abstractmethod
    def mode_dimensions(self):
        """
        :returns: list of local Hilbert space dimensions
        :rtype: list[int]
        """
        pass

    @abstractmethod
    def dim(self, i):
        """Returns the local Hilbert space dimension of mode i

        :param i: The index of the mode
        :type i: int

        :returns: local Hilbert space dimension of mode i
        :rtype: int
        """
        pass

    @abstractmethod
    def nmodes(self):
        """
        :returns: The number of modes in the multiset TTN
        :rtype: int
        """
        pass

    @abstractmethod
    def is_purification(self):
        """
        :returns: Whether or not the state represents a purification
        :rtype: bool
        """
        pass

    @abstractmethod
    def ntensors(self):
        """
        :returns: The total number of tensors in the tensor network
        :rtype: int
        """
        pass

    @abstractmethod
    def nsites(self):
        """
        :returns: The total number of tensors in the tensor network
        :rtype: int
        """
        pass

    @abstractmethod
    def nset(self):
        """
        :returns: The number of set variables for the multiset TTN.  Here it is one
        :rtype: int
        """
        pass

    @abstractmethod
    def nelems(self):
        """
        :returns: The total number of elements in all tensors of the network.
        :rtype: int
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        :returns: The number of modes in the multiset TTN
        :rtype: int
        """
        pass

    # def compute_maximum_bond_entropy(self):
    #    """Computes the maximum SvN across any bond in the tensor network and returns the results

    #    :returns: The maximum bond entropy in the tensor network
    #    :rtype: float

    #    """
    #    pass

    # def maximum_bond_entropy(self):
    #    """Returns the previously computed maximum SvN across any bond in the tensor network

    #    :returns: The maximum bond entropy in the tensor network
    #    :rtype: float

    #    """
    #    pass

    # def bond_entropy(self, i):
    #    """Returns the SvN across the ith bond of the current orthogonality centre.
    #    Where for all nodes but the root 0 corresponds to the parent of the current orthogonality centre and its children are then 1-nchild,
    #    For the root i just indexes the children

    #    :returns: The bond entropy
    #    :rtype: float

    #    """
    #    pass

    # def maximum_bond_dimension(self):
    #    """
    #    :returns: The maximum bond dimension
    #    :rtype: int

    #    """
    #    pass

    # def minimum_bond_dimension(self):
    #    """
    #    :returns: The minimum bond dimension
    #    :rtype: int

    #    """
    #    pass

    @abstractmethod
    def has_orthogonality_centre(self):
        """            
        :returns: Whether or not the multiset TTN has an active orthogonality centre
        :rtype: bool

        """
        pass

    @abstractmethod
    def orthogonality_centre(self):
        """            
        :returns: The index of the current orthogonality centre
        :rtype: int

        """
        pass

    @abstractmethod
    def is_orthogonalised(self):
        """            
        :returns: Whether or not the multiset TTN has an orthogonality centre at the root
        :rtype: bool

        """
        pass

    @abstractmethod 
    def force_set_orthogonality_centre(self, i):
        """Sets the orthogonality centre of the tensor network to index i but does not modify the tensor to ensure that this is a
        valid orthogonality centre

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i: int or list[int]

        """
        pass

    @abstractmethod   
    def shift_orthogonality_centre(self, i, tol=0, nchi=0):
        """Shift the orthogonality centre down the ith bond of the current orthogonality centre with possible truncation. 
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

    @abstractmethod
    def set_orthogonality_centre(self, i, tol=0, nchi=0):
        """Sets the orthogonality centre of the tensor network to index i either introducing an orthogonality centre if there is none
        or simply shifting the orthogonality centre from its current location to the required location

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i: int or list[int]
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    @abstractmethod
    def orthogonalise(self, force=False):
        """Shifts the orthogonality centre to the root node of the multiset TTN

        :param force: Whether or not to force a full reorthogonalisation of the multiset TTN regardless of whether or not it believes it has an orthogonality centre
        :type force: bool, optional

        """
        pass

    @abstractmethod    
    def truncate(self, tol=0, nchi=0):
        """Ensures the tensor network is in an orthogonalised form.  Then performs an euler tour truncating each bond according to the user
        specified tol and nchi parameters

        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    @abstractmethod
    def normalise(self):
        """Ensures the multiset TTN is a normalised to one and returns the previous value of the norm of the tensor

        :returns: The previous 2-norm of the multiset TTN
        :rtype: float
        """
        pass

    @abstractmethod
    def norm(self):
        """
        :returns: The 2-norm of the multiset TTN
        :rtype: float
        """
        pass

    @abstractmethod
    def __setitem__(self, i, v):
        r"""Sets the value of a site tensor in the tensor network

        :param i: Index of the node to set
        :type i: int
        :param v: The new value of the node data object
        :type v: multiset_ttn_data

        """
        pass

    @abstractmethod
    def __getitem__(self, i, v):
        """Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :returns: tensor data
        :rtype: multiset_ttn_data

        """
        pass

    # def set_site_tensor(self, i, v):
    #    """Sets the value of a site tensor in the tensor network

    #    :param i: Index of the node to set
    #    :type i: int
    #    :param v: The new value of the node data object
    #    :type v: ttn_data

    #    """
    #    pass

    @abstractmethod
    def site_tensor(self, i, v):
        r"""Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :returns: tensor data
        :rtype: multiset_ttn_data

        """
        pass

    # def measure_without_collapse(self, i):
    #    """Evaluate the probablity of observing each state following a projective measurement applied to mode i without performing the collapse

    #    :param i: The physical mode to perform the projective measurement on
    #    :type i: int

    #    :returns: The probability of observing each basis state following the projective measurement
    #    :rtype: list[float]
    #    """
    #    pass

    # def collapse_basis(self, U, truncate=True, tol=0, nchi=0):
    #    """Perform a projective measurement across all modes in the multiset TTN applying a basis transformation U_i to each mode i before doing so

    #    :param U: A list of basis transformations to apply to the state before performing the projective measurement
    #    :type U: list[np.ndarray] or list[linalg.matrix]
    #    :param truncate: Whether or not to truncate the state following collapse as it is a product state. (Default: True)
    #    :type truncate: bool, optional
    #    :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
    #    :type tol: float, optional
    #    :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
    #    :type nchi: int, optional

    #    :returns: The probability of this collapse event occurint
    #    :rtype: float
    #    """
    #    pass

    # def collapse(self, truncate=True, tol=0, nchi=0):
    #    """Perform a projective measurement across all modes in the multiset TTN

    #    :param truncate: Whether or not to truncate the state following collapse as it is a product state. (Default: True)
    #    :type truncate: bool, optional
    #    :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
    #    :type tol: float, optional
    #    :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
    #    :type nchi: int, optional

    #    :returns: The probability of this collapse event occurint
    #    :rtype: float
    #    """
    #    pass

    # def apply_one_body_operator(self, *args, shift_orthogonality=True):
    #    """Apply a one-body operator to the multiset TTN updating its value

    #    :param *args: A variable length list of arguments. Valid options are
    #
    #        - **op** (linalg.matrix or np.ndarray or site_operator), **mode** (int) -  Apply the operator op to mode mode
    #        - **op** (site_operator) - Apply the operator op to the mode specified by op
    #    :type *args: [Arguments (variable number and type)]
    #    :param shift_orthogonality: Whether or not to shift the orthogonality centre of the multiset TTN to the leaf node that will be updated by this one-body operator.  (Default: True)
    #    :type shift_orthogonality: bool, optional
    #    """
    #    pass

    # def apply_product_operator(self, op, shift_orthogonality=True):
    #    """Apply a product of one-body operator to the multiset TTN updating its value

    #    :param op: The product operator to apply to the system
    #    :type op: product_operator
    #    :param shift_orthogonality: Whether or not to shift the orthogonality centre of the multiset TTN to the leaf node that will be updated by this one-body operator.  (Default: True)
    #    :type shift_orthogonality: bool, optional
    #    """
    #    pass

    # def apply_operator(self, op, shift_orthogonality=True):
    #    """Apply a product of one-body operator to the multiset TTN updating its value

    #    :param op: The product operator to apply to the system
    #    :type op: site_operator or product_operator
    #    :param shift_orthogonality: Whether or not to shift the orthogonality centre of the multiset TTN to the leaf node that will be updated by this one-body operator.  (Default: True)
    #    :type shift_orthogonality: bool, optional
    #    """
    #    pass

    # def __imatmul__(self, op):
    #    """Apply an operator to the multiset TTN updating its value.  Shifting the orthogonality centre to the leaf nodes that will be updated by this operator

    #    :param op: The product operator to apply to the system
    #    :type op: site_operator or product_operator or sop_opertor

    #    """
    #    pass

    # def __rmatmul__(A,  op):
    #    """Apply an operator to the multiset TTN updating its value, returning the result as a new multiset TTN

    #    :param A: The multiset ttn to apply the operator to
    #    :type A: ms_ttn
    #    :param op: The product operator to apply to the system
    #    :type op: site_operator or product_operator or sop_opertor

    #    :returns: The result of op@A
    #    :rtype: ms_ttn

    #    """
    #    pass

multiset_ttn.register(ms_ttn_complex)
if _real_ttn_import:
    multiset_ttn.register(ms_ttn_real)
if _cuda_import:
    multiset_ttn.register(ms_ttn_complex_cuda)
    if _real_ttn_import:
       multiset_ttn.register(ms_ttn_real_cuda)

ms_ttn = multiset_ttn
ms_ttn_type = multiset_ttn
multiset_ttn_type = multiset_ttn

