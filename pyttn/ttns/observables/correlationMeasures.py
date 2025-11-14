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

import numpy as np
from typing import Optional, Union, Tuple
from scipy.special import xlogy

from ..ttns.ttnExt import ttn
from .rdmExt import rdm


#function for computing logNegativity 
class CorrelationMeasures:
    def __init__(self):
        """A class for computing two body entanglement measures of a TTN object.
        """
        self._rdm = None


    def __compute_rdms(self, A : ttn, i : int, j : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Function for computing the required RDM objects for evaluating two body entanglement measures

        :param A: The tree tensor network object
        :type A: ttn
        :param i: The first mode to consider.
        :type i: int
        :param j: The second mode to consider.
        :type j: int
        :return: The two-body and two one-body rdms needed for computing two-body entanglement measures
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        rdm_ij = self._rdm(A, i, j)
        rdm_i = self._rdm(A, i)
        rdm_j = self._rdm(A, j)
        return np.array(rdm_ij), np.array(rdm_i), np.array(rdm_j)

    def __compute_log_negativity(self, A : ttn, i : int, j : int) -> float:
        """Compute the log negativity between two modes in a TTN

        :param A: The tree tensor network object
        :type A: ttn
        :param i: The first mode to consider.
        :type i: int
        :param j: The second mode to consider.
        :type j: int
        :return: The log negativity between modes i and j 
        :rtype: float
        """

        rdm_ij, rdm_i, rdm_j = self.__compute_rdms(A, i, j)

    #def logNegativity(self, A : ttn, i : Optional[int] = None, j : Optional[int] = None) -> Union[float | np.ndarray]:
    #    """Compute the log negativity between two modes in a TTN or the matrix of log negativities between all modes
    #
    #    :param A: The tree tensor network object
    #    :type A: ttn
    #    :param i: The first mode to consider.  If none we compute the full matrix of log Negativities, defaults to None
    #    :type i: int, optional
    #    :param j: The second mode to consider.  If none we compute the full matrix of log Negativities, defaults to None
    #    :type j: int, optional
    #    :return: Either the log negativity between modes i and j or the matrix of log negativities between all modes
    #    :rtype: Union[float | np.ndarray]
    #    """
    #    self.__init_rdm_engine(A)
    #    
    #    #if either i or j are none 
    #    if i is None or j is None:
    #        res = np.zeros((len(A), len(A)))
    #        for _i in range(len(A)):
    #            for _j in range(i+1, len(A)):
    #                res[_i, _j] = self.__compute_log_negativity(A, _i, _j)
    #                res[_j, _i] = res[_i, _j]
    #        return res
    #    else:
    #        return self.__compute_log_negativity(A, i, j)


    def __von_neumann_entropy(rho : np.ndarray) -> float:
        """Compute the von neumann entropy for a density operator
        ..math::
            S(\\rho) = -tr(\\rho \\mathrm{ln}(\\rho))

        :param rho: Density operator
        :type rho: np.ndarray
        :return: The value of the von Neumann entropy
        :rtype: float
        """

        li = np.linalg.eigvalsh(rho)
        li = np.where(li < 0, 0, li)
        return -np.sum(xlogy(li, li))


    def __compute_mutual_information(self, A : ttn, i : int, j : int) -> float:
        """Compute the log negativity between two modes in a TTN

        :param A: The tree tensor network object
        :type A: ttn
        :param i: The first mode to consider.
        :type i: int
        :param j: The second mode to consider.
        :type j: int
        :return: The log negativity between modes i and j 
        :rtype: float
        """

        rdm_ij, rdm_i, rdm_j = self.__compute_rdms(A, i, j)
        Sij = CorrelationMeasures.__von_neumann_entropy(rdm_ij)
        Si = CorrelationMeasures.__von_neumann_entropy(rdm_i)
        Sj = CorrelationMeasures.__von_neumann_entropy(rdm_j)
        print(Si, Sj, Sij)
        return Si + Sj - Sij


    def mutualInformation(self, A : ttn, i : Optional[int] = None, j : Optional[int] = None) -> Union[float | np.ndarray]:
        """Compute the Mutual Information between two modes in a TTN or the matrix of Mutual Information between all modes

        :param A: The tree tensor network object
        :type A: ttn
        :param i: The first mode to consider.  If none we compute the full matrix of Mutual Information, defaults to None
        :type i: int, optional
        :param j: The second mode to consider.  If none we compute the full matrix of Mutual Information, defaults to None
        :type j: int, optional
        :return: Either the Mutual Information between modes i and j or the matrix of Mutual Information between all modes
        :rtype: Union[float | np.ndarray]
        """
        self.__init_rdm_engine(A)
        
        #if either i or j are none 
        if i is None or j is None:
            res = np.zeros((len(A), len(A)))
            for _i in range(len(A)):
                for _j in range(_i+1, len(A)):
                    res[_i, _j] = self.__compute_mutual_information(A, _i, _j)
                    res[_j, _i] = res[_i, _j]
            return res
        else:
            return self.__compute_mutual_information(A, i, j)


    def __init_rdm_engine(self, A : ttn):
        if self._rdm is None:
            self._rdm = rdm(A, two_body=True)
        else:
            self._rdm.resize(A, two_body=True)
        
