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
from typing import Iterator, Optional, Union

import numpy as np

from pyttn.linalg import Matrix
from pyttn.ttnpp import ms_ttn_complex, ms_ttn_node_complex
from .ttnExt import ttnNodeData

# and attempt to import the real ttns
try:
    from pyttn.ttnpp import ms_ttn_real, ms_ttn_node_real

    _real_ttn_import = True
except ImportError:
    _real_ttn_import = False

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import ms_ttn_complex as ms_ttn_complex_cuda
    from pyttn.ttnpp.cuda import ms_ttn_node_complex as ms_ttn_node_complex_cuda

    _cuda_import = True
    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import ms_ttn_real as ms_ttn_real_cuda
        from pyttn.ttnpp.cuda import ms_ttn_node_real as ms_ttn_node_real_cuda

except ImportError:
    _cuda_import = False



def is_ms_ttn_node(A) -> bool:
    """A function for determining whether a given object is a msttnNode object
    :param A: The object to test
    :return: Whether or not the object is a ttnNode
    :rtype: bool
    """
    ret = isinstance(A, ms_ttn_node_complex)
    if _real_ttn_import:
        ret = ret or isinstance(A, ms_ttn_node_real)

    if _cuda_import:
        ret = ret or isinstance(A, ms_ttn_node_complex_cuda)

        if _real_ttn_import:
            ret = ret or isinstance(A, ms_ttn_node_real_cuda)

    return ret

def _ms_ttn_node_blas(dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
        return ms_ttn_node_complex()
    elif dtype == np.float64 or dtype is float:
        return ms_ttn_node_real()
    else:
        raise RuntimeError("Invalid dtype for msttnNode")



def _ms_ttn_node_cuda(dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
        return ms_ttn_node_complex_cuda()
    elif dtype == np.float64 or dtype is float:
        return ms_ttn_node_real_cuda()
    else:
        raise RuntimeError("Invalid dtype for msttnNode")


class msttnNode(metaclass=ABCMeta):
    """Class for handling a node in a ttn"""
    def __new__(
        cls, 
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas"
    ) -> "msttnNode":
        """Factory function for constructing  a tree tensor network node

        :param dtype: The dtype to use for the ttn node.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the ttn node.  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :return: The Tree Tensor Network Node object
        :rtype: msttnNode
        """
        if backend == "blas":
            return _ms_ttn_node_blas(dtype=dtype)
        elif _cuda_import and backend == "cuda":
            return _ms_ttn_node_cuda(dtype=dtype)
        else:
            raise RuntimeError("Invalid backend type for msttnNode")
        
    @abstractmethod
    def data(self) -> list[ttnNodeData]:
        """Returns the ttnNodeData object stored at the node

        :return: The tensor node information stored at the node
        :rtype: list[ttnNodeData]
        """
        pass

    @abstractmethod
    def __call__(self) -> list[ttnNodeData]:
        """Returns the ttnNodeData object stored at the node

        :return: The tensor node information stored at the node
        :rtype: list[ttnNodeData]
        """
        pass

    @abstractmethod
    def is_root(self) -> bool:
        """Returns whether or not the current node is the root

        :return: Whether or not the current node is the root
        :rtype: bool
        """
        pass

    @abstractmethod
    def is_leaf(self) -> bool:
        """Returns whether or not the current node is a leaf

        :return: Whether or not the current node is a leaf
        :rtype: bool
        """
        pass
    
    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the object stores a complex valued dtype

        :return: dtype
        :rtype: bool
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns the NumPy dtype of the underlying ttn representation.

        This corresponds to the scalar type used internally, e.g.
        ``np.float64`` or ``np.complex128``.

        :return: The dtype of the operator
        :rtype: numpy.dtype
        """
        pass

    @abstractmethod
    def conj(self) -> None:
        "Take the complex conjugate of the msttnNode.  Here this is evaluated lazily"
        pass    

    @abstractmethod
    def nmodes(self) -> int:
        """
        :return: The number of modes stored at this node.  That is the number of children of the node.
        :rtype: int
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: The number of modes stored at this node.  That is the number of children of the node.
        :rtype: int
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        :return: A string represent of the msttnNode object
        :rtype: str
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns a string labelling the backend of the msttnNode object

        :return: backend label
        :rtype: str
        """
        pass

    @abstractmethod
    def __getitem__(self, ind : int) -> "msttnNode":
        """Returns the child of the current node at index ind

        :param ind: The index of the child to access
        :type ind: int
        :return: The child at index ind
        :rtype: msttnNode
        """
        pass

    @abstractmethod
    def child(self, ind : int) -> "msttnNode":
        """Returns the child of the current node at index ind

        :param ind: The index of the child to access
        :type ind: int
        :return: The child at index ind
        :rtype: msttnNode
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        :return: Iterator object over the children of the current node
        :rtype: Iterator
        """
        pass

msttnNode.register(ms_ttn_node_complex)
if _real_ttn_import:
    msttnNode.register(ms_ttn_node_real)
if _cuda_import:
    msttnNode.register(ms_ttn_node_complex_cuda)
    if _real_ttn_import:
        msttnNode.register(ms_ttn_node_real_cuda)

ms_ttn_node = msttnNode
multiset_ttn_node = msttnNode
def is_ms_ttn(A) -> bool:
    """A function for determining whether a given object is a multiset ttn
    :param A: The object to test
    :return: Whether or not the object is a ms_ttn
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
    :return: Whether or not the object is a ms_ttn
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





class msttn(metaclass=ABCMeta):
    """Class for handling multiset tree tensor network state operator
    """   
    def __new__(cls, *args,
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = np.complex128,
        backend: str = "blas",
        **kwargs,
    ) -> 'msttn':
        """Factory function for constructing a multiset tree tensor network state operator

        :param `*args`: Variable length list of arguments. This function can handle two possible lists of arguments

            - Default construct a multiset TTN object
            - msttn (:class:`msttn`) - Copy construct a multiset TTN object
            - tree (:class:`ntree`), nset (int) - Construct a multiset TTN from an Ntree object and the number of set variables
            - string (str), nset (int) - Construct multiset TTN from an string defining an Ntree object and the number of set variables

        :param dtype: The dtype to use for the ttn.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the ttn.  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :param `**kwargs`: Additional keyword arguments that are based to the ttn object constructor
        :return: The Multiset Tree Tensor Network State object
        :rtype: msttn 
        """   
        if backend == "blas":
            return _ms_ttn_blas(*args, dtype=dtype, **kwargs)
        elif _cuda_import and backend == "cuda":
            return _ms_ttn_cuda(*args, dtype=dtype, **kwargs)
        else:
            raise RuntimeError("Invalid backend type for msttn")


    @abstractmethod
    def complex_dtype(self) -> bool:
        """Return the data type stored in the multiset ttn

        :return: Whether or not the object has a complex dtype
        :rtype: bool
        """
        pass
    
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns the NumPy dtype of the underlying ttn representation.

        This corresponds to the scalar type used internally, e.g.
        ``np.float64`` or ``np.complex128``.

        :return: The dtype of the operator
        :rtype: numpy.dtype
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns a string labelling the backend of the multiset ttn object

        :return: backend label
        :rtype: str
        """
        pass

    @abstractmethod
    def nthreads(self) -> int:
        """Stores the number of threads that can be used to attempt to parallelise updates over the set variables

        :return: dtype
        :rtype: {np.complex128 or np.float64}
        """
        pass

    @abstractmethod    
    def assign(self, o) -> None:
        """Assign the value of this multiset ttn from another multiset ttn

        :param o: The other multiset ttn object
        :type o: msttn
        """
        pass

    @abstractmethod
    def slice(self, i) -> "msttnSlice":  # type: ignore # noqa: F821
        """Returns a slice object that allows for easy accessing of the multiset ttn correspond to a single set variable i

        :param i: The set index of the slice
        :type i: int

        :return: A slice of the multiset ttn used for accessing the tensor network associated with a single system state
        :rtype: "msttnSlice"

        """

    @abstractmethod
    def bond(self):
        """Return a list of all bonds in the network

        :return: All bonds in the network
        :rtype: list[int, int]
        """
        pass

    @abstractmethod
    def bond_dimensions(self) -> dict[tuple[int,int], list[int]]:
        """Return a dictionary containing the bond (the two sites forming the bond) and bond dimension of all bonds in the network

        :return: All bond dimensions in the network
        :rtype: dict[tuple[int,int], list[int]]
        """
        pass

    @abstractmethod
    def bond_capacities(self) -> dict[tuple[int,int], list[int]]:
        """Return a dictionary containing the bond (the two sites forming the bond) and maximum bond dimension of all bonds in the network

        :return: All maximum bond dimensions in the network
        :rtype: dict[tuple[int,int], list[int]]
        """
        pass

    @abstractmethod
    def reset_orthogonality_centre(self) -> None:
        """Resets the orthogonality centre of the multiset TTN to the root node of the tree."""
        pass

    @abstractmethod
    def resize(self, *args, nset, purification=False) -> None:
        """Resize the multiset TTN object given a new set of topology information. This optionally takes a flag allowing for the state to automatically represent a purification of a wavefunction

        :param `*args`: A variable length list of arguments. Valid options are

            - **topology** (:class:`ntree` or str) - Construct a multiset ttn from a ntree object defining the topology and bond dimensions of the ttn
            - **topology** (:class:`ntree` or str), **capacity** (ntree or str ) - Construct a multiset ttn from an ntree object defining the topology and a capacity defining the maximum bond dimensions
        :type `*args`: [Arguments (variable number and type)]
        :param nset: The number of set variables to use for the msttn
        :type nset: int
        :param purification: Whether or not the buffers should be resized to store a purification of the requested state size.  (Default: False)
        :type purification: bool, optional
        """
        pass

    @abstractmethod
    def set_seed(self, seed) -> None:
        """Set the value of the random number generate seed used for internal operations requiring random sampling

        :param seed: The new value of the seed
        :type seed: int
        """
        pass

    @abstractmethod
    def set_state(self, *args, random_unoccupied_initialisation=False) -> None:
        """Set the coefficients in the multiset TTN so that it represents a user specified product state

        :param `*args`: A variable length list of arguments. Valid options are

            - **set_index** (int), **state** (list[int]) - For setting the state to be a product state vector acting on a specific set index
            - **coeff** (list[dtype]), **state** (list[list[int]]) - For setting the system to be the state :math:`\\sum_i c_i \\|i\\rangle \\bigotimes \\mathrm{state}_i`

        :param random_unoccupied_initialisation: Whether or not to set all other elements of the multiset TTN not determining the product state to random values or not. (Default: False)
        :type random_unoccupied_initialisation: bool, optional
        """
        pass

    # def set_product(self, state):
    #    """Set the coefficients in the multiset TTN so that it represents a product of a set of one body states

    #    :param `*args`:`:`: A variable length list of arguments. Valid options are
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
    def __imul__(self, b : Union[float, complex, np.float64, np.complex128]) -> "msttn":
        """Inplace multiplication of the multiset TTN object by a scalar

        :param b: Scalar value to multiply multiset TTN by
        :type b: Union[float, complex, np.float64, np.complex128]
        :return: The result of the inplace multiplication
        :rtype: msttn
        """
        pass

    @abstractmethod
    def __idiv__(self, b : Union[float, complex, np.float64, np.complex128]) -> "msttn":
        """Inplace division of the multiset TTN object by a scalar

        :param b: Scalar value to divide multiset TTN by
        :type b: Union[float, complex, np.float64, np.complex128]
        :return: The result of the inplace division
        :rtype: msttn
        """
        pass

    @abstractmethod
    def conj(self) -> None:
        "Take the complex conjugate of the multiset TTN.  Here this is evaluated lazily"
        pass

    @abstractmethod
    def random(self) -> None:
        "Sample the coefficients in the multiset TTN randomly from a normal distribution"
        pass

    @abstractmethod
    def zero(self) -> None:
        "Set all coefficients in the multiset TTN to zero"
        pass

    @abstractmethod    
    def clear(self) -> None:
        "Clear and deallocate all internal buffers of the multiset TTN"
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        :return: Iterator object over nodes in multiset TTN
        :rtype: Iterator
        """
        pass

    @abstractmethod
    def mode_dimensions(self) -> list[int]:
        """
        :return: list of local Hilbert space dimensions
        :rtype: list[int]
        """
        pass

    @abstractmethod
    def dim(self, i : int) -> int:
        """Returns the local Hilbert space dimension of mode i

        :param i: The index of the mode
        :type i: int

        :return: local Hilbert space dimension of mode i
        :rtype: int
        """
        pass

    @abstractmethod
    def nmodes(self) -> int:
        """
        :return: The number of modes in the multiset TTN
        :rtype: int
        """
        pass

    @abstractmethod
    def is_purification(self) -> bool:
        """
        :return: Whether or not the state represents a purification
        :rtype: bool
        """
        pass

    @abstractmethod
    def ntensors(self) -> int:
        """
        :return: The total number of tensors in the tensor network
        :rtype: int
        """
        pass

    @abstractmethod
    def nsites(self) -> int:
        """
        :return: The total number of tensors in the tensor network
        :rtype: int
        """
        pass

    @abstractmethod
    def nset(self) -> int:
        """
        :return: The number of set variables for the multiset TTN.
        :rtype: int
        """
        pass

    @abstractmethod
    def nelems(self) -> int:
        """
        :return: The total number of elements in all tensors of the network.
        :rtype: int
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: The number of modes in the multiset TTN
        :rtype: int
        """
        pass

    # def compute_maximum_bond_entropy(self):
    #    """Computes the maximum SvN across any bond in the tensor network and returns the results

    #    :return: The maximum bond entropy in the tensor network
    #    :rtype: float

    #    """
    #    pass

    # def maximum_bond_entropy(self):
    #    """Returns the previously computed maximum SvN across any bond in the tensor network

    #    :return: The maximum bond entropy in the tensor network
    #    :rtype: float

    #    """
    #    pass

    # def bond_entropy(self, i):
    #    """Returns the SvN across the ith bond of the current orthogonality centre.
    #    Where for all nodes but the root 0 corresponds to the parent of the current orthogonality centre and its children are then 1-nchild,
    #    For the root i just indexes the children

    #    :return: The bond entropy
    #    :rtype: float

    #    """
    #    pass

    # def maximum_bond_dimension(self):
    #    """
    #    :return: The maximum bond dimension
    #    :rtype: int

    #    """
    #    pass

    # def minimum_bond_dimension(self):
    #    """
    #    :return: The minimum bond dimension
    #    :rtype: int

    #    """
    #    pass

    @abstractmethod
    def has_orthogonality_centre(self) -> bool:
        """            
        :return: Whether or not the multiset TTN has an active orthogonality centre
        :rtype: bool

        """
        pass

    @abstractmethod
    def orthogonality_centre(self) -> int:
        """            
        :return: The index of the current orthogonality centre
        :rtype: int

        """
        pass

    @abstractmethod
    def is_orthogonalised(self) -> bool:
        """            
        :return: Whether or not the multiset TTN has an orthogonality centre at the root
        :rtype: bool

        """
        pass

    @abstractmethod 
    def force_set_orthogonality_centre(self, i : Union[int, list[int]]) -> None:
        """Sets the orthogonality centre of the tensor network to index i but does not modify the tensor to ensure that this is a
        valid orthogonality centre

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i:  Union[int, list[int]]

        """
        pass

    @abstractmethod   
    def shift_orthogonality_centre(self, i : int, tol : float = 0, nchi : int =0) -> None:
        """Shift the orthogonality centre down the ith bond of the current orthogonality centre with possible truncation. 
        Where for all nodes but the root 0 corresponds to the parent of the current orthogonality centre and its children are then 1-nchild
        For the root i just indexes the children

        :param i: The index of the bond of the current node that we will shift the orthogonality centre across
        :type i: int
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    @abstractmethod
    def set_orthogonality_centre(self, i  : Union[int, list[int]] , tol : float =0, nchi : int =0) -> None:
        """Sets the orthogonality centre of the tensor network to index i either introducing an orthogonality centre if there is none
        or simply shifting the orthogonality centre from its current location to the required location

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i: Union[int, list[int]]
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    @abstractmethod
    def orthogonalise(self, force : bool =False) -> None:
        """Shifts the orthogonality centre to the root node of the multiset TTN

        :param force: Whether or not to force a full reorthogonalisation of the multiset TTN regardless of whether or not it believes it has an orthogonality centre
        :type force: bool, optional

        """
        pass

    @abstractmethod    
    def truncate(self, tol : float = 0, nchi : int =0) -> None:
        """Ensures the tensor network is in an orthogonalised form.  Then performs an euler tour truncating each bond according to the user
        specified tol and nchi parameters

        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        """
        pass

    @abstractmethod
    def normalise(self) -> float:
        """Ensures the multiset TTN is a normalised to one and returns the previous value of the norm of the tensor

        :return: The previous 2-norm of the multiset TTN
        :rtype: float
        """
        pass

    @abstractmethod
    def norm(self) -> float:
        """
        :return: The 2-norm of the multiset TTN
        :rtype: float
        """
        pass

    @abstractmethod
    def __setitem__(self, i, v : list[ttnNodeData]) -> None:
        """Sets the value of a site tensor in the tensor network

        :param i: Index of the node to set
        :type i: int
        :param v: The new value of the node data object
        :type v: list[ttnNodeData]
        """
        pass

    @abstractmethod
    def __getitem__(self, i) -> list[ttnNodeData]:
        """Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :return: tensor data
        :rtype: list[ttnNodeData]

        """
        pass

    @abstractmethod
    def set_site_tensor(self, i : int, v : Union[list[Matrix], list[np.ndarray]]) -> None:
        """Sets the value of a site tensor in the tensor network

        :param i: Index of the node to set
        :type i: int
        :param v: The new value of the node data object
        :type v: Union[list[Matrix], list[np.ndarray]]

        """
        pass

    @abstractmethod
    def site_tensor(self, i) -> list[Matrix]:
        """Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :return: tensor data
        :rtype: list[Matrix]

        """
        pass

    # def measure_without_collapse(self, i):
    #    """Evaluate the probablity of observing each state following a projective measurement applied to mode i without performing the collapse

    #    :param i: The physical mode to perform the projective measurement on
    #    :type i: int

    #    :return: The probability of observing each basis state following the projective measurement
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

    #    :return: The probability of this collapse event occurint
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

    #    :return: The probability of this collapse event occurint
    #    :rtype: float
    #    """
    #    pass

    # def apply_one_body_operator(self, *args, shift_orthogonality=True):
    #    """Apply a one-body operator to the multiset TTN updating its value

    #    :param `*args`: A variable length list of arguments. Valid options are
    #
    #        - **op** (linalg.matrix or np.ndarray or site_operator), **mode** (int) -  Apply the operator op to mode mode
    #        - **op** (site_operator) - Apply the operator op to the mode specified by op
    #    :type `*args`: [Arguments (variable number and type)]
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
    #    :type A: msttn
    #    :param op: The product operator to apply to the system
    #    :type op: site_operator or product_operator or sop_opertor

    #    :return: The result of op@A
    #    :rtype: msttn

    #    """
    #    pass

    @abstractmethod
    def save(self, fname : str, as_binary: bool = True):
        """Serialise the msTTN object to a file fname.  

        :param fname: The output file name
        :type fname: str
        :param as_binary: Whether or not to save as a binary file, defaults to True
        :type as_binary: bool, optional

        """
        pass

    @abstractmethod
    def load(self, fname : str, as_binary: bool = True):
        """Load a msTTN object from the file fname.  
        
        :param fname: The input file name
        :type fname: str
        :param as_binary: Whether or not to load as a binary file, defaults to True
        :type as_binary: bool, optional

        """
        pass

msttn.register(ms_ttn_complex)
if _real_ttn_import:
    msttn.register(ms_ttn_real)
if _cuda_import:
    msttn.register(ms_ttn_complex_cuda)
    if _real_ttn_import:
       msttn.register(ms_ttn_real_cuda)

ms_ttn = msttn
multiset_ttn = msttn

