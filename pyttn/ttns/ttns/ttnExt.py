# This files is part of the pyttn package.
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
from typing import Iterator, Optional, Union

import numpy as np

from pyttn.linalg import Matrix
from pyttn.ttnpp import ttn_complex, ttn_node_complex, ttn_data_complex
from pyttn.ttns.operators.opExt import Op
# and attempt to import the real ttns
try:
    from pyttn.ttnpp import ttn_real, ttn_node_real, ttn_data_real

    _real_ttn_import = True
except ImportError:
    _real_ttn_import = False

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import ttn_complex as ttn_complex_cuda
    from pyttn.ttnpp.cuda import ttn_node_complex as ttn_node_complex_cuda
    from pyttn.ttnpp.cuda import ttn_data_complex as ttn_data_complex_cuda

    _cuda_import = True
    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import ttn_real as ttn_real_cuda
        from pyttn.ttnpp.cuda import ttn_node_real as ttn_node_real_cuda
        from pyttn.ttnpp.cuda import ttn_data_real as ttn_data_real_cuda
except ImportError:
    _cuda_import = False

def available_backends() -> list[str]:
    if _cuda_import:
        return ["blas", "cuda"]
    else:
        return ["blas"]

def is_ttn_node_data(A) -> bool:
    """A function for determining whether a given object is a ttnNodeData object
    :param A: The object to test
    :return: Whether or not the object is a ttnNodeData
    :rtype: bool
    """
    ret = isinstance(A, ttn_data_complex)
    if _real_ttn_import:
        ret = ret or isinstance(A, ttn_data_real)

    if _cuda_import:
        ret = ret or isinstance(A, ttn_data_complex_cuda)

        if _real_ttn_import:
            ret = ret or isinstance(A, ttn_data_real_cuda)

    return ret

def _ttn_data_blas(*args, dtype=np.complex128):
    if args:
        if isinstance(args[0], ttn_data_complex):
            return ttn_data_complex(*args)
        elif _real_ttn_import and isinstance(args[0], ttn_data_real):
            if dtype == np.complex128 or dtype is complex:
                return ttn_data_complex(*args)
            else:
                return ttn_data_real(*args)
        else:
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                return ttn_data_complex(*args)
            elif dtype == np.float64 or dtype is float:
                return ttn_data_real(*args)
            else:
                raise RuntimeError("Invalid dtype for ttnNodeData")
    else:
        if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
            return ttn_data_complex(*args)
        elif dtype == np.float64 or dtype is float:
            return ttn_data_real(*args)
        else:
            raise RuntimeError("Invalid dtype for ttnNodeData")


def _ttn_data_cuda(*args, dtype=np.complex128):
    if args:
        if isinstance(args[0], ttn_data_complex_cuda):
            return ttn_data_complex_cuda(*args)
        elif _real_ttn_import and isinstance(args[0], ttn_data_real_cuda):
            if dtype == np.complex128 or dtype is complex:
                return ttn_data_complex_cuda(*args)
            else:
                return ttn_data_real_cuda(*args)
        else:
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                return ttn_data_complex_cuda(*args)
            elif dtype == np.float64 or dtype is float:
                return ttn_data_real_cuda(*args)
            else:
                raise RuntimeError("Invalid dtype for ttnNodeData")
    else:
        if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
            return ttn_data_complex_cuda(*args)
        elif dtype == np.float64 or dtype is float:
            return ttn_data_real_cuda(*args)
        else:
            raise RuntimeError("Invalid dtype for ttnNodeData")


class ttnNodeData(metaclass=ABCMeta):
    """Class for handling the data stored in a ttn node object"""
    def __new__(
        cls, 
        *args, 
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas"
    ) -> "ttnNodeData":
        """Factory function for constructing the data object stored in a tree tensor network
    
        :param `*args`: Variable length list of arguments. This function can handle the following lists of arguments

            - Default construct ttnNodeData object
            - data (:class:`ttnNodeData`) - Copy construct ttnNodeData object

        :param dtype: The dtype to use for the ttnNodeData.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the ttnNodeData.  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :return: The Tree Tensor Network Node Data object
        :rtype: ttnNodeData
        """
        if backend == "blas":
            return _ttn_data_blas(*args, dtype=dtype)
        elif _cuda_import and backend == "cuda":
            return _ttn_data_cuda(*args, dtype=dtype)
        else:
            raise RuntimeError("Invalid backend type for ttnNodeData")

    @abstractmethod
    def resize(self, hrank: int, mode_dims: list[int]) -> None:
        """Resize the ttnNodeData object so it has a maximum upwards pointing bond dimension of hrank 
        and dimensions mode_dims pointing towards its children

        :param hrank: The upwards pointing bond dimension (or alternatively the number of single particle functions associated with the node)
        :type hrank: int
        :param mode_dims: An array of downward pointing bond dimensions
        :type mode_dims: list[int]
        """
        pass
    
    @abstractmethod
    def reallocate(self, max_hrank: int, max_mode_dims: list[int]) -> None:
        """Reallocate the buffer stored in the ttnNodeData object so it can allow for a maximum upwards pointing bond dimension of max_hrank 
        and maximum dimensions max_mode_dims pointing towards its children

        :param max_hrank: The maximum upwards pointing bond dimension (or alternatively the number of single particle functions associated with the node)
        :type max_hrank: int
        :param max_mode_dims: An array of the maximum downward pointing bond dimensions
        :type max_mode_dims: list[int]
        """
        pass     

    @abstractmethod
    def is_orthogonalised(self) -> bool:
        """Stores whether the node is an isometry.  That is whether it has been orthogonalised so that the orthogonality centre is above the current node. 

        :return: Whether or not the data stored in this node is an isometry
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

    @abstractmethod
    def conj(self) -> None:
        "Take the complex conjugate of the ttnNodeData.  Here this is evaluated lazily"
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
    def hrank(self) -> int:
        """
        :return: The upwards pointing bond dimension of the node (or alternatively the number of single particle functions)
        :rtype: int
        """
        pass

    @abstractmethod
    def nspf(self) -> int:
        """
        :return: The number of single particle functions (or alternatively the upwards pointing bond dimension of the node)
        :rtype: int
        """
        pass

    @abstractmethod
    def dimen(self, use_max_dim: bool = False) -> int:
        """Returns the number of basis states that this node acts on.  That is the product of the child nodes upwards pointing bond dimensions

        :param use_max_dim: Whether or not to return the maximum dimension of the mode, defaults to False
        :type use_max_dim: bool, optional
        :return: The product of the child nodes upwards pointing bond dimensions
        :rtype: int
        """
        pass

    @abstractmethod
    def dim(self, i : int) -> int:
        """
        :param i: The index of the child to consider
        :type i: int
        :return: The number of single particle functions associated with the ith child of this node
        :rtype: int
        """
        pass

    @abstractmethod
    def dims() -> list[int]:
        """
        :return: A list containing the number of single particle functions associated with each child of this node
        :rtype: list[int]
        """
        pass

    @abstractmethod
    def set_dim(self, i : int, n : int) -> None:
        """Set the new value of the number of basis states associated with child i
        :param i: The index of the child to consider
        :type i: int
        :param n: The new number of basis states
        :type n: int
        """
        pass

    @abstractmethod
    def nelems(self) -> int:
        """
        :return: The total number of elements in the node tensor
        :rtype: int
        """
        pass

    @abstractmethod
    def nset(self) -> int:
        """
        :return: The number of set variables handled by the node.  Here this is alwasy 1
        :rtype: int
        """
        pass

    @abstractmethod
    def max_hrank(self) -> int:
        """
        :return: The maximum allowed upwards pointing bond dimension of the node (or alternatively the number of single particle functions)
        :rtype: int
        """
        pass

    @abstractmethod
    def max_nspf(self) -> int:
        """
        :return: The maximum allowed number of single particle functions (or alternatively the upwards pointing bond dimension of the node)
        :rtype: int
        """
        pass

    @abstractmethod
    def max_dim(self, i : int) -> int:
        """
        :param i: The index of the child to consider
        :type i: int
        :return: The maximum allowed number of single particle functions associated with the ith child of this node
        :rtype: int
        """
        pass

    @abstractmethod
    def max_dims() -> list[int]:
        """
        :return: A list containing the maximum allowed number of single particle functions associated with each child of this node
        :rtype: list[int]
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        "Clear and deallocate all internal buffers of the ttnNodeData object"
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        :return: A string represent of the ttnNodeData object
        :rtype: str
        """
        pass

    @abstractmethod
    def set_matrix(self, mat : Union[Matrix, np.ndarray]) -> None:
        """Set the value of the tensor stored at this node so that its matricisation where the first index corresponds to the upwards
        pointing bond and the second index is the direct product of the downward pointing bonds  

        :param mat: The new value of the node tensor
        :type mat: Union[Matrx, np.ndarray]
        """
        pass

    @abstractmethod
    def as_matrix(self) -> Matrix:
        """Returns the matricisation of the current tensor, where the first index corresponds to the upwards
        pointing bond and the second index is the direct product of the downward pointing bonds

        :return: Matricisation of the current tensor
        :rtype: Matrix
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns a string labelling the backend of the ttnNodeData object

        :return: backend label
        :rtype: str
        """
        pass


ttnNodeData.register(ttn_data_complex)
if _real_ttn_import:
    ttnNodeData.register(ttn_data_real)
if _cuda_import:
    ttnNodeData.register(ttn_data_complex_cuda)
    if _real_ttn_import:
        ttnNodeData.register(ttn_data_real_cuda)

ttn_node_data = ttnNodeData

def is_ttn_node(A) -> bool:
    """A function for determining whether a given object is a ttnNode object
    :param A: The object to test
    :return: Whether or not the object is a ttnNode
    :rtype: bool
    """
    ret = isinstance(A, ttn_node_complex)
    if _real_ttn_import:
        ret = ret or isinstance(A, ttn_node_real)

    if _cuda_import:
        ret = ret or isinstance(A, ttn_node_complex_cuda)

        if _real_ttn_import:
            ret = ret or isinstance(A, ttn_node_real_cuda)

    return ret

def _ttn_node_blas(dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
        return ttn_node_complex()
    elif dtype == np.float64 or dtype is float:
        return ttn_node_real()
    else:
        raise RuntimeError("Invalid dtype for ttnNode")



def _ttn_node_cuda(dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
        return ttn_node_complex_cuda()
    elif dtype == np.float64 or dtype is float:
        return ttn_node_real_cuda()
    else:
        raise RuntimeError("Invalid dtype for ttnNode")


class ttnNode(metaclass=ABCMeta):
    """Class for handling a node in a ttn"""
    def __new__(
        cls, 
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas"
    ) -> "ttnNode":
        """Factory function for constructing  a tree tensor network node

        :param dtype: The dtype to use for the ttn node.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the ttn node.  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :return: The Tree Tensor Network Node Data object
        :rtype: ttnNode
        """
        if backend == "blas":
            return _ttn_node_blas(dtype=dtype)
        elif _cuda_import and backend == "cuda":
            return _ttn_node_cuda(dtype=dtype)
        else:
            raise RuntimeError("Invalid backend type for ttnNode")
        
    @abstractmethod
    def data(self) -> ttnNodeData:
        """Returns the ttnNodeData object stored at the node

        :return: The tensor node information stored at the node
        :rtype: ttnNodeData
        """
        pass

    @abstractmethod
    def __call__(self) -> ttnNodeData:
        """Returns the ttnNodeData object stored at the node

        :return: The tensor node information stored at the node
        :rtype: ttnNodeData
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

    @abstractmethod
    def conj(self) -> None:
        "Take the complex conjugate of the ttnNode.  Here this is evaluated lazily"
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
        :return: A string represent of the ttnNode object
        :rtype: str
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns a string labelling the backend of the ttnNode object

        :return: backend label
        :rtype: str
        """
        pass

    @abstractmethod
    def __getitem__(self, ind : int) -> "ttnNode":
        """Returns the child of the current node at index ind

        :param ind: The index of the child to access
        :type ind: int
        :return: The child at index ind
        :rtype: ttnNode
        """
        pass

    @abstractmethod
    def child(self, ind : int) -> "ttnNode":
        """Returns the child of the current node at index ind

        :param ind: The index of the child to access
        :type ind: int
        :return: The child at index ind
        :rtype: ttnNode
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        :return: Iterator object over the children of the current node
        :rtype: Iterator
        """
        pass

ttnNode.register(ttn_node_complex)
if _real_ttn_import:
    ttnNode.register(ttn_node_real)
if _cuda_import:
    ttnNode.register(ttn_node_complex_cuda)
    if _real_ttn_import:
        ttnNode.register(ttn_node_real_cuda)

ttn_node = ttnNode

def is_ttn(A) -> bool:
    """A function for determining whether a given object is a ttn
    :param A: The object to test
    :return: Whether or not the object is a ttn
    :rtype: bool
    """
    ret = isinstance(A, ttn_complex)
    if _real_ttn_import:
        ret = ret or isinstance(A, ttn_real)

    if _cuda_import:
        ret = ret or isinstance(A, ttn_complex_cuda)

        if _real_ttn_import:
            ret = ret or isinstance(A, ttn_real_cuda)

    return ret

def _ttn_blas(*args, dtype=np.complex128, **kwargs):
    if args:
        if isinstance(args[0], ttn_complex):
            return ttn_complex(*args, **kwargs)
        elif _real_ttn_import and isinstance(args[0], ttn_real):
            if dtype == np.complex128 or dtype is complex:
                return ttn_complex(*args, **kwargs)
            else:
                return ttn_real(*args, **kwargs)
        else:
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                return ttn_complex(*args, **kwargs)
            elif dtype == np.float64 or dtype is float:
                return ttn_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ttn")
    else:
        if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
            return ttn_complex(*args, **kwargs)
        elif dtype == np.float64 or dtype is float:
            return ttn_real(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ttn")


def _ttn_cuda(*args, dtype=np.complex128, **kwargs):
    if args:
        if isinstance(args[0], ttn_complex_cuda):
            return ttn_complex_cuda(*args, **kwargs)
        elif _real_ttn_import and isinstance(args[0], ttn_real_cuda):
            if dtype == np.complex128 or dtype is complex:
                return ttn_complex_cuda(*args, **kwargs)
            else:
                return ttn_real_cuda(*args, **kwargs)
        else:
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                return ttn_complex_cuda(*args, **kwargs)
            elif dtype == np.float64 or dtype is float:
                return ttn_real_cuda(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ttn")
    else:
        if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
            return ttn_complex_cuda(*args, **kwargs)
        elif dtype == np.float64 or dtype is float:
            return ttn_real_cuda(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ttn")



class ttn(metaclass=ABCMeta):
    """Class for handling a tree tensor network state object
    """
    def __new__(
        cls,
        *args,
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas",
        **kwargs,
    ) -> "ttn":
        """Factory function for constructing a tree tensor network state object
    
        :param `*args`: Variable length list of arguments. This function can handle the following lists of arguments

            - Default construct ttn object
            - ttn (:class:`ttn`) - Copy construct ttn object
            - slice (:class:`ms_ttn_slice`) - Construct ttn object from slice of multiset ttn
            - tree (:class:`ntree`) - Construct ttn from an Ntree object
            - string (str) - Construct ttn from an string defining an Ntree object

        :param dtype: The dtype to use for the ttn.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the ttn.  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :param `**kwargs`: Additional keyword arguments that are based to the ttn object constructor
        :return: The Tree Tensor Network State object
        :rtype: ttn
        """
        if backend == "blas":
            return _ttn_blas(*args, dtype=dtype, **kwargs)
        elif _cuda_import and backend == "cuda":
            return _ttn_cuda(*args, dtype=dtype, **kwargs)
        else:
            raise RuntimeError("Invalid backend type for ttn")

    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the object stores a complex valued dtype

        :return: dtype
        :rtype: bool
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns a string labelling the backend of the ttn object

        :return: backend label
        :rtype: str
        """
        pass

    @abstractmethod
    def assign(self, o : 'ttn') -> None:
        """Assign the value of this ttn from another ttn

        :param o: The other ttn object
        :type o: ttn
        """
        pass

    @abstractmethod
    def bond(self) -> list[tuple[int, int]]:
        """Return a list of all bonds in the network

        :return: All bonds in the network
        :rtype: list[tuple[int, int]]
        """
        pass

    @abstractmethod
    def bond_dimensions(self) -> dict[tuple[int, int], int]:
        """Return a dictionary containing the bond (the two sites forming the bond) and bond dimension of all bonds in the network

        :return: All bond dimensions in the network
        :rtype: dict[tuple[int, int], int]
        """
        pass

    @abstractmethod
    def bond_capacities(self) -> dict[tuple[int, int], int]:
        """Return a dictionary containing the bond (the two sites forming the bond) and maximum bond dimension of all bonds in the network

        :return: All maximum bond dimensions in the network
        :rtype: dict[tuple[int, int], int]
        """
        pass

    @abstractmethod
    def reset_orthogonality_centre(self) -> None:
        """Resets the orthogonality centre of the ttn to the root node of the tree."""
        pass

    @abstractmethod
    def resize(self, *args, purification: bool =False) -> None:
        """Resize the ttn object given a new set of topology information. This optionally takes a flag allowing for the state to automatically represent a purification of a wavefunction

        :param `*args`: A variable length list of arguments. Valid options are

            - **topology** (:class:`ntree` or str) - Construct a ttn from a ntree object defining the topology and bond dimensions of the ttn
            - **topology** (:class:`ntree` or str), **capacity** (ntree or str ) - Construct a ttn from an ntree object defining the topology and a capacity defining the maximum bond dimensions
        
        :type `*args`: [Arguments (variable number and type)]
        :param purification: Whether or not the buffers should be resized to store a purification of the requested state size.  (Default: False)
        :type purification: bool, optional
        """
        pass

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set the value of the random number generate seed used for internal operations requiring random sampling

        :param seed: The new value of the seed
        :type seed: int
        """
        pass

    @abstractmethod
    def set_state(self, state: list[int], random_unoccupied_initialisation: bool=False) -> None:
        """Set the coefficients in the ttn so that it represents a user specified product state

        :param state: The occupation number state to set the ttn to
        :type state: list[int]
        :param random_unoccupied_initialisation: Whether or not to set all other elements of the ttn not determining the product state to random values or not. (Default: False)
        :type random_unoccupied_initialisation: bool, optional
        """
        pass

    @abstractmethod
    def set_product(self, state : list[list[Union[float, complex, np.float64, np.complex128]]]) -> None:
        """Set the coefficients in the ttn so that it represents a product of a set of one body states

        :param state: A list containing a set of vectors corresponding to the individual product states
        :type state: list[list[Union[float, complex, np.float64, np.complex128]]]
        """
        pass

    @abstractmethod
    def set_identity_purification(self) -> None:
        """Sets the state of the ttn to a purification state representing the identity"""
        pass

    @abstractmethod
    def sample_product(self, dist: list[list[Union[float, complex, np.float64, np.complex128]]]) -> None:
        """Sample a direct product of occupation states from a set of probabilities of observing each mode in a given state

        :param state: A list containing a set of vectors corresponding to the probabilities of observing each occupation state
        :type state: list[list[Union[float, complex, np.float64, np.complex128]]]
        """
        pass

    @abstractmethod
    def __imul__(self, b : Union[float, complex, np.float64, np.complex128]) -> "ttn":
        """Inplace multiplication of the ttn object by a scalar

        :param b: Scalar value to multiply ttn by
        :type b: Union[float, complex, np.float64, np.complex128]
        :return: The result of the inplace multiplication
        :rtype: ttn
        """
        pass

    @abstractmethod
    def __idiv__(self, b : Union[float, complex, np.float64, np.complex128]) -> "ttn":
        """Inplace division of the ttn object by a scalar

        :param b: Scalar value to divide ttn by
        :type b: Union[float, complex, np.float64, np.complex128]
        :return: The result of the inplace division
        :rtype: ttn
        """
        pass

    @abstractmethod
    def conj(self) -> None:
        "Take the complex conjugate of the ttn.  Here this is evaluated lazily"
        pass

    @abstractmethod
    def random(self) -> None:
        "Sample the coefficients in the ttn randomly from a normal distribution"
        pass

    @abstractmethod
    def zero(self) -> None:
        "Set all coefficients in the ttn to zero"
        pass

    @abstractmethod
    def clear(self) -> None:
        "Clear and deallocate all internal buffers of the ttn"
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        :return: Iterator object over nodes in ttn
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
    def dim(self, i: int) -> int:
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
        :return: The number of modes in the ttn
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
        :return: The number of set variables for the ttn.  Here it is one
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
        :return: The number of modes in the ttn
        :rtype: int
        """
        pass

    @abstractmethod
    def compute_maximum_bond_entropy(self) -> float:
        """Computes the maximum SvN across any bond in the tensor network and returns the results

        :return: The maximum bond entropy in the tensor network
        :rtype: float

        """
        pass

    @abstractmethod
    def maximum_bond_entropy(self) -> float:
        """Returns the previously computed maximum SvN across any bond in the tensor network

        :return: The maximum bond entropy in the tensor network
        :rtype: float

        """
        pass

    @abstractmethod
    def bond_entropy(self, i: int) -> float:
        """Returns the SvN across the ith bond of the current orthogonality centre.
        Where for all nodes but the root 0 corresponds to the parent of the current orthogonality centre and its children are then 1-nchild,
        For the root i just indexes the children

        :return: The bond entropy
        :rtype: float

        """
        pass

    @abstractmethod
    def maximum_bond_dimension(self) -> int:
        """
        :return: The maximum bond dimension
        :rtype: int

        """
        pass

    @abstractmethod
    def minimum_bond_dimension(self) -> int:
        """
        :return: The minimum bond dimension
        :rtype: int

        """
        pass

    @abstractmethod
    def has_orthogonality_centre(self) -> bool:
        """
        :return: Whether or not the ttn has an active orthogonality centre
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
        :return: Whether or not the ttn has an orthogonality centre at the root
        :rtype: bool

        """
        pass

    @abstractmethod
    def force_set_orthogonality_centre(self, i : Union[int, list[int]]) -> None:
        """Sets the orthogonality centre of the tensor network to index i but does not modify the tensor to ensure that this is a
        valid orthogonality centre

        :param i: The index of or a list of ints defining the traversal path to reach the node correspond to the new orthogonality centre
        :type i: Union[int, list[int]]

        """
        pass

    @abstractmethod
    def shift_orthogonality_centre(self, i: int, tol: float=0, nchi: int=0) -> None:
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
    def set_orthogonality_centre(self, i : Union[int, list[int]], tol: float=0, nchi: int=0) -> None:
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
    def orthogonalise(self, force: bool=False) -> None:
        """Shifts the orthogonality centre to the root node of the ttn

        :param force: Whether or not to force a full reorthogonalisation of the ttn regardless of whether or not it believes it has an orthogonality centre
        :type force: bool, optional

        """

    @abstractmethod
    def truncate(self, tol: float=0, nchi: int=0) -> None:
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
        """Ensures the ttn is a normalised to one and returns the previous value of the norm of the tensor

        :return: The previous 2-norm of the ttn
        :rtype: float
        """
        pass

    @abstractmethod
    def norm(self) -> float:
        """
        :return: The 2-norm of the ttn
        :rtype: float
        """
        pass

    @abstractmethod
    def __setitem__(self, i : int, v : ttnNodeData):
        """Sets the value of a site tensor in the tensor network

        :param i: Index of the node to set
        :type i: int
        :param v: The new value of the node data object
        :type v: ttnNodeData

        """
        pass

    @abstractmethod
    def __getitem__(self, i : int) ->  ttnNodeData:
        """Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :return: tensor data
        :rtype: ttnNodeData

        """
        pass

    @abstractmethod
    def at(self, i : int) ->  ttnNodeData:
        """Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :return: tensor data
        :rtype: ttnNodeData

        """
        pass

    @abstractmethod
    def node(self, i : int) ->  ttnNode:
        """Access ttn Node at index i

        :param i: Index of the node to access data from
        :type i: int

        :return: ttn Node
        :rtype: ttnNode

        """
        pass

    @abstractmethod
    def set_site_tensor(self, i : int, v : Union[Matrix, np.ndarray] ) -> None:
        """Sets the value of a site tensor in the tensor network

        :param i: Index of the node to set
        :type i: int
        :param v: The new value of the node data object
        :type v: Union[Matrix, np.ndarray]

        """
        pass

    @abstractmethod
    def site_tensor(self, i : int) -> Matrix:
        """Access tensor data at node i

        :param i: Index of the node to access data from
        :type i: int

        :return: tensor data
        :rtype: Matrix

        """
        pass

    @abstractmethod
    def measure_without_collapse(self, i : int) -> list[float]:
        """Evaluate the probablity of observing each state following a projective measurement applied to mode i without performing the collapse

        :param i: The physical mode to perform the projective measurement on
        :type i: int

        :return: The probability of observing each basis state following the projective measurement
        :rtype: list[float]
        """
        pass

    @abstractmethod
    def collapse_basis(self, U: Union[list[np.ndarray], list[Matrix]] , truncate: bool=True, tol: float=0, nchi: int=0) -> float:
        """Perform a projective measurement across all modes in the ttn applying a basis transformation U_i to each mode i before doing so

        :param U: A list of basis transformations to apply to the state before performing the projective measurement
        :type U: Union[list[np.ndarray], list[Matrix]]
        :param truncate: Whether or not to truncate the state following collapse as it is a product state. (Default: True)
        :type truncate: bool, optional
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        :return: The probability of this collapse event occurint
        :rtype: float
        """
        pass

    @abstractmethod
    def collapse(self, truncate: bool=True, tol: float=0, nchi: int=0) -> float:
        """Perform a projective measurement across all modes in the ttn

        :param truncate: Whether or not to truncate the state following collapse as it is a product state. (Default: True)
        :type truncate: bool, optional
        :param tol: A truncation tolerance for the singular values to discard weight.  (Default: 0)
        :type tol: float, optional
        :param nchi: A maximum bond dimension to truncate to.  This is ignored if nchi=0.  (Default: 0)
        :type nchi: int, optional

        :return: The probability of this collapse event occurint
        :rtype: float
        """
        pass

    @abstractmethod
    def apply_one_body_operator(self, *args, shift_orthogonality: bool=True) -> None:
        """Apply a one-body operator to the ttn updating its value

        :param `*args`: A variable length list of arguments. Valid options are

            - **op** (:class:`Matrix` or np.ndarray or :class:`site_operator`), **mode** (int) -  Apply the operator op to mode mode
            - **op** (:class:`site_operator`) - Apply the operator op to the mode specified by op

        :param shift_orthogonality: Whether or not to shift the orthogonality centre of the ttn to the leaf node that will be updated by this one-body operator.  (Default: True)
        :type shift_orthogonality: bool, optional
        """
        pass

    @abstractmethod
    def apply_product_operator(self, op, shift_orthogonality: bool=True) -> None:
        """Apply a product of one-body operator to the ttn updating its value

        :param op: The product operator to apply to the system
        :type op: product_operator
        :param shift_orthogonality: Whether or not to shift the orthogonality centre of the ttn to the leaf node that will be updated by this one-body operator.  (Default: True)
        :type shift_orthogonality: bool, optional
        """
        pass

    @abstractmethod
    def apply_op(self, op : Op, tol : float = -1.0, nchi : int = 0, zipup : bool = False) -> None:
        """
        Apply a matrix valued operator object to the ttn updating its value

        :param op: The product operator to apply to the system
        :type op: Op
        :param tol: The truncation tolerance, defaults to -1.0
        :type tol: float, optional
        :param nchi: The maximum bond dimension, defaults to 0
        :type nchi: int, optional
        :param zipup: Whether or not to use the zipup algorithm for computing the action of the operator on the ttn, defaults to False
        :type zipup: bool, optional
        """
        pass

    @abstractmethod
    def apply_operator(self, op, shift_orthogonality: bool=True) -> None:
        """Apply a product of one-body operator to the ttn updating its value

        :param op: The product operator to apply to the system
        :type op: site_operator or product_operator
        :param shift_orthogonality: Whether or not to shift the orthogonality centre of the ttn to the leaf node that will be updated by this one-body operator.  (Default: True)
        :type shift_orthogonality: bool, optional
        """
        pass

    @abstractmethod
    def __imatmul__(self, op) -> "ttn":
        """Apply an operator to the ttn updating its value.  Shifting the orthogonality centre to the leaf nodes that will be updated by this operator

        :param op: The product operator to apply to the system
        :type op: site_operator or product_operator or sop_opertor
        :return: The result of the inplace operation op@A
        :rtype: ttn
        """
        pass

    @abstractmethod
    def __rmatmul__(self, op) -> 'ttn':
        """Apply an operator to the ttn updating its value, returning the result as a new ttn

        :param op: The product operator to apply to the system
        :type op: site_operator or product_operator or sop_opertor

        :return: The result of op@self
        :rtype: ttn

        """
        pass

    @abstractmethod
    def save(self, fname : str, as_binary: bool = True):
        """Serialise the TTN object to a file fname.  

        :param fname: The output file name
        :type fname: str
        :param as_binary: Whether or not to save as a binary file, defaults to True
        :type as_binary: bool, optional

        """
        pass

    @abstractmethod
    def load(self, fname : str, as_binary: bool = True):
        """Load a TTN object from the file fname.  
        
        :param fname: The input file name
        :type fname: str
        :param as_binary: Whether or not to load as a binary file, defaults to True
        :type as_binary: bool, optional

        """
        pass

ttn.register(ttn_complex)
if _real_ttn_import:
    ttn.register(ttn_real)
if _cuda_import:
    ttn.register(ttn_complex_cuda)
    if _real_ttn_import:
        ttn.register(ttn_real_cuda)
