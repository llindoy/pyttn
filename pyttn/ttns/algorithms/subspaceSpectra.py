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
from typing import Union 
import numpy as np

from pyttn.ttns.operators.sopOperatorExt import MultisetSOPOperator, SOPOperator

class SubspaceEngine(metaclass=ABCMeta):
    def __init__(self):
        self._vectors = []
        self._Hp = None
        self._Sp = None
        self._B = None
        self._A = None

        self._canonical = False

        self._Hco = None
        self._Co = None

    def canonical_orthogonalisation(self, eps : float = 1e-12):
        if not isinstance(self._S, np.ndarray):
            raise RuntimeError("Cannot perform canonical orthogonalisation.  Subspace vectors have not been constructed.")
        
        #consturct a regularised canonical orbitals.  Here we do this by discarding orbitals with SVD less than float
        u, v = np.linalg.eigh(self._S)

        Ntrunc = np.sum( (u > eps).sum())

        u = np.where( u  < eps, 0, 1/np.sqrt(u))
        D = np.diag(u)
        U = v

        D = D[:, :Ntrunc]
        self._Co = U@D
        self._Hco = np.conj(self._Co).T@self._Hp@self.Co
        self._canonical = True
        
    def evaluate_spectra(self):
        #ensure that the system has been canonically orthogonalised
        if not self._canonical:
            self.canonical_orthogonalisation()

        #now compute the eigenvalues and eigenvectors of the Hamiltonian in the canonically orthogonalised space
        w, phi = np.linalg.eigh(self._Hco)
            