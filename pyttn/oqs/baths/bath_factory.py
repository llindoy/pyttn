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

from typing import Callable, Optional, Union

import numpy as np

from pyttn.ttns import OP_type

from ..spectral_density import CorrelatedSpectralDensity
from .bosonic_bath import BosonicBath
from .correlated_bosonic_bath import CorrelatedBosonicBath
from .discretised_bath import (
    DiscreteBosonicBath,
    DiscreteFermionicBath,
    DiscreteOQSBath,
)
from .discretised_correlated_bath import DiscreteCorrelatedBosonicBath
from .fermionic_bath import FermionicBath


def bosonic_bath(
    Jw: Union[Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]], CorrelatedSpectralDensity],
    S: Optional[Union[OP_type, list[OP_type]]] = None,
    beta: Optional[float] = None,
    wmax: float = np.inf,
    wmin: Optional[float] = None,
) -> Union[BosonicBath, CorrelatedBosonicBath]:
    """A factory method for constructing a bosonic bath object

    :param Jw: The bath spectral function defining the non-interacting correlation function
    :type Jw: Union[Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]], CorrelatedSpectralDensity]
    :param S: The system operator
    :type S: OP_type, optional
    :param beta: The inverse temperature of the bath, defaults to None
    :type beta: float, optional
    :param wmax: the maximum frequency bound, defaults to np.inf
    :type wmax: float, optional
    :param wmin: the minimum frequency bound, defaults to None
    :type wmin: float, optional

    :returns: The bosonic bath
    :rtype: Union[BosonicBath, CorrelatedBosonicBath]
    """

    if isinstance(Jw, CorrelatedSpectralDensity):
        return CorrelatedBosonicBath(Jw, S=S, beta=beta, wmax=wmax, wmin=wmin)
    elif callable(Jw):
        return BosonicBath(Jw, S=S, beta=beta, wmax=wmax, wmin=wmin)
    else:
        raise RuntimeError(
            "Failed to create bosonic bath. Failed to recognised type of bath spectral density."
        )


def fermionic_bath(
    Jw: Union[Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]], CorrelatedSpectralDensity],
    Sp: Optional[Union[OP_type, list[OP_type]]] = None,
    Sm: Optional[Union[OP_type, list[OP_type]]] = None,
    beta: Optional[float] = None,
    wmax: float = np.inf,
    wmin: Optional[float] = None,
    wtol: Optional[float] = None,
) -> FermionicBath:
    """A factory method for constructing a fermionic bath object

    :param Jw: The bath spectral function defining the non-interacting correlation function
    :type Jw: Union[Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]], CorrelatedSpectralDensity]
    :param Sp: The system raising operators
    :type Sp: Optional[OP_type]
    :param Sm: The system raising operators
    :type Sm: Optional[OP_type]
    :param beta: The inverse temperature of the bath, defaults to None
    :type beta: float, optional
    :param wmax: the maximum frequency bound, default to np.inf
    :type wmax: float, optional
    :param wmin: the minimum frequency bound, default to np.inf
    :type wmin: float, optional
    :param wtol: a value for determining wmin based on wmax.  See fermionic.bath.estimate_bounds, default to None
    :type wtol: float, optional

    :returns: The fermionic bath
    :rtype: FermionicBath
    """

    if isinstance(Jw, CorrelatedSpectralDensity):
        raise RuntimeError("Correlated fermionic bath not yet supported.")
        # return CorrelatedBosonicBath(Jw, S=S, beta=beta, wmax=wmax, wmin=wmin)
    elif callable(Jw):
        return FermionicBath(
            Jw, Sp=Sp, Sm=Sm, beta=beta, wmax=wmax, wmin=wmin, wtol=wtol
        )
    else:
        raise RuntimeError(
            "Failed to create bosonic bath. Failed to recognised type of bath spectral density."
        )


def discrete_bosonic_bath(
    gk: np.ndarray, wk: np.ndarray, tol: float = 1e-12
) -> Union[DiscreteBosonicBath, DiscreteCorrelatedBosonicBath]:
    """A factory method for constructing a discrete bosonic bath objects

    :param gk: The coefficient in the exponential decomposition
    :type gk: np.ndarrays
    :param wk: The decay rates in the exponential decomposition
    :type wk: np.ndarray
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional

    :returns: The correct bosonic bath object
    :rtype: Union[DiscreteBosonicBath, DiscreteCorrelatedBosonicBath]
    """
    if gk.ndim == 1:
        return DiscreteBosonicBath(gk, wk, tol=tol)
    elif gk.ndim == 3:
        return DiscreteCorrelatedBosonicBath(gk, wk, tol=tol)
    else:
        raise RuntimeError("Failed to determine which type of bosonic bath to return.")


def discrete_fermionic_bath(
    gk: np.ndarray, wk: np.ndarray, tol: float = 1e-12
) -> DiscreteFermionicBath:
    """A factory method for constructing a discrete bosonic bath objects

    :param gk: The coefficient in the exponential decomposition
    :type gk: np.ndarrays
    :param wk: The decay rates in the exponential decomposition
    :type wk: np.ndarray
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional

    :returns: The correct fermionic bath object
    :rtype: DiscreteFermionicBath
    """
    if gk.ndim == 1:
        return DiscreteFermionicBath(gk, wk, tol=tol)
    else:
        raise RuntimeError("Failed to determine which type of bosonic bath to return.")


def discrete_bath(
    gk: np.ndarray, wk: np.ndarray, fermionic: bool = False, tol: float = 1e-12
) -> DiscreteOQSBath:
    """A factory method for constructing a discrete bath objects

    :param gk: The coefficient in the exponential decomposition
    :type gk: np.ndarrays
    :param wk: The decay rates in the exponential decomposition
    :type wk: np.ndarray
    :param fermionic: Whether or not the bath is fermionic, defaults to False
    :type fermionic: bool
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional

    :returns: The correct bath object
    :rtype: DiscreteOQSBath
    """
    if fermionic:
        return discrete_fermionic_bath(gk, wk, tol=tol)
    else:
        return discrete_bosonic_bath(gk, wk, tol=tol)
