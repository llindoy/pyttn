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
import numpy as np

from pyttn.ttnpp import ms_ttn_slice_complex

# and attempt to import the real ttns
try:
    from pyttn.ttnpp import ms_ttn_slice_real

    _real_ttn_import = True
except ImportError:
    _real_ttn_import = False


# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import ttn_complex as ms_ttn_slice_complex_cuda

    _cuda_import = True
    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import ttn_real as ms_ttn_slice_real_cuda

except ImportError:
    _cuda_import = False

from .msttnExt import is_ms_ttn
from .ttnExt import ttn


def is_ms_ttn_slice(A) -> bool:
    """A function for determining whether a given object is a multiset ttn
    :param A: The object to test
    :returns: Whether or not the object is a ms_ttn
    :rtype: bool
    """
    ret = isinstance(A, ms_ttn_slice_complex)
    if _real_ttn_import:
        ret = ret or isinstance(A, ms_ttn_slice_real)

    if _cuda_import:
        ret = ret or isinstance(A, ms_ttn_slice_complex_cuda)

        if _real_ttn_import:
            ret = ret or isinstance(A, ms_ttn_slice_real_cuda)

    return ret

def _msttn_slice_1(slice):
    if is_ms_ttn_slice(slice):
        if slice.complex_dtype():
            if slice.backend() == "blas":
                return ms_ttn_slice_complex(slice)
            elif slice.backend() == "cuda" and _cuda_import:
                return ms_ttn_slice_complex_cuda(slice)
            else:
                raise RuntimeError("Invalid arguments for msttn_slice_1")
        elif _real_ttn_import:    
            if slice.backend() == "blas":
                return ms_ttn_slice_real(slice)
            elif slice.backend() == "cuda":
                return ms_ttn_slice_real_cuda(slice)      
            else:
                raise RuntimeError("Invalid arguments for msttn_slice_1")
    else:
        raise RuntimeError("Invalid arguments for msttn_slice_1")
    
def _msttn_slice_2(mstree, ind):
    try:
        if is_ms_ttn(mstree) and isinstance(ind, int):
            if mstree.complex_dtype():
                if mstree.backend() == "blas":
                    return ms_ttn_slice_complex(mstree, ind)
                elif mstree.backend() == "cuda" and _cuda_import:
                    return ms_ttn_slice_complex_cuda(mstree, ind)
                else:
                    raise RuntimeError("Invalid arguments for msttn_slice_1")
            elif _real_ttn_import:    
                if mstree.backend() == "blas":
                    return ms_ttn_slice_real(mstree, ind)
                elif mstree.backend() == "cuda":
                    return ms_ttn_slice_real_cuda(mstree, ind)      
                else:
                    raise RuntimeError("Invalid arguments for msttn_slice_1")
        else:
            raise RuntimeError("Invalid arguments for msttn_slice_1")
    except Exception as ex:
        raise ex

def _msttn_slice_1(slice):
    if is_ms_ttn(slice):
        if slice.complex_dtype():
            if slice.backend() == "blas":
                return ms_ttn_slice_complex(slice)
            elif slice.backend() == "cuda" and _cuda_import:
                return ms_ttn_slice_complex_cuda(slice)
            else:
                raise RuntimeError("Invalid arguments for msttn_slice_1")
        elif _real_ttn_import:    
            if slice.backend() == "blas":
                return ms_ttn_slice_real(slice)
            elif slice.backend() == "cuda":
                return ms_ttn_slice_real_cuda(slice)      
            else:
                raise RuntimeError("Invalid arguments for msttn_slice_1")
    else:
        raise RuntimeError("Invalid arguments for msttn_slice_1")

class msttnSlice(metaclass=ABCMeta):
    """Class for handling multiset tree tensor network state operator
    """   
    def __new__(cls, *args) -> 'msttnSlice':
        """Factory function for constructing a multiset tree tensor network state operator

        :param `*args`: Variable length list of arguments. This function can handle two possible lists of arguments

            - Default construct a multiset TTN object
            - slice (:class:`msttnSlice`), index (int) - Copy construct a multiset TTN slice object
            - msttn (:class:`multiset_ttn`), index (int) - Construct a multiset TTN slice as a slice of a given multiset ttn

        :returns: The Multiset Tree Tensor Network State Slice object
        :rtype: `msttnSlice`
        """   
        if len(args) == 1:
            return _msttn_slice_1(args[0])
        elif len(args) == 2:
            return _msttn_slice_2(args[0], args[1])
        else:
            raise RuntimeError("Invalid arguments for msttnSlice constructor.")

    @abstractmethod
    def assign(self, tree : ttn):
        """Assign the value in the multiset ttn slice object from a ttn object

        :param tree: ttn
        :type tree: ttn
        """
        pass   
    
    @abstractmethod
    def complex_dtype(self):
        """Return the data type stored in the multiset ttn slice

        :returns: dtype
        :rtype: {np.complex128 or np.float64}
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
        """Returns a string labelling the backend of the multiset ttn slice object

        :returns: backend label
        :rtype: str
        """
        pass

    @abstractmethod
    def nset(self):
        """
        :returns: The number of set variables for the multiset TTN slice.  Here it is one
        :rtype: int
        """
        pass

msttnSlice.register(ms_ttn_slice_complex)
if _real_ttn_import:
    msttnSlice.register(ms_ttn_slice_real)
if _cuda_import:
    msttnSlice.register(ms_ttn_slice_complex_cuda)
    if _real_ttn_import:
       msttnSlice.register(ms_ttn_slice_real_cuda)

ms_ttn_slice = msttnSlice
multiset_ttn_slice = msttnSlice
