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

from pyttn.ttnpp import (
    adaptive_one_site_tdvp_complex,
    multiset_one_site_tdvp_complex,
    one_site_tdvp_complex,
)
from pyttn.ttns.operators.mssopOperatorExt import ms_sop_operator
from pyttn.ttns.operators.sopOperatorExt import sop_operator
from pyttn.ttns.ttns.msttnExt import ms_ttn
from pyttn.ttns.ttns.ttnExt import ttn

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import (
        mulitset_one_site_tdvp_complex as multiset_one_site_tdvp_complex_cuda,
    )
    from pyttn.ttnpp.cuda import one_site_tdvp_complex as one_site_tdvp_complex_cuda

    _cuda_import = True

except ImportError:
    _cuda_import = False


def _one_site_tdvp_blas(A, H, **kwargs):
    if isinstance(A, ttn) and isinstance(H, sop_operator):
        return one_site_tdvp_complex(A, H, **kwargs)
    elif isinstance(A, ms_ttn) and isinstance(H, ms_sop_operator):
        return multiset_one_site_tdvp_complex(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid inputs for one site tdvp")


def _one_site_tdvp_cuda(A, H, **kwargs):
    if isinstance(A, ttn) and isinstance(H, sop_operator):
        return one_site_tdvp_complex_cuda(A, H, **kwargs)
    elif isinstance(A, ms_ttn) and isinstance(H, ms_sop_operator):
        return multiset_one_site_tdvp_complex_cuda(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid inputs for one site tdvp")


def _one_site_tdvp(A, H, **kwargs):
    if A.backend() == "blas" and H.backend() == "blas":
        return _one_site_tdvp_blas(A, H, **kwargs)
    elif A.backend() == "cuda" and H.backend() == "cuda":
        return _one_site_tdvp_cuda(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid backend arguments.")


def _subspace_tdvp_blas(A, H, **kwargs):
    if isinstance(A, ttn) and isinstance(H, sop_operator):
        return adaptive_one_site_tdvp_complex(A, H, **kwargs)
    elif isinstance(A, ms_ttn) and isinstance(H, ms_sop_operator):
        raise RuntimeError(
            "Subspace expansion based integrator has not yet been implemented for Multiset TTNs"
        )
    else:
        raise RuntimeError("Invalid inputs for one site tdvp")


def _subspace_tdvp_cuda(A, H, **kwargs):
    if isinstance(A, ttn) and isinstance(H, sop_operator):
        raise RuntimeError(
            "Subspace expansion based integrator has not yet been implemented for CUDA"
        )
    elif isinstance(A, ms_ttn) and isinstance(H, ms_sop_operator):
        raise RuntimeError(
            "Subspace expansion based integrator has not yet been implemented for Multiset TTNs"
        )
    else:
        raise RuntimeError("Invalid inputs for one site tdvp")


def _subspace_tdvp(A, H, **kwargs):
    if A.backend() == "blas":
        return _subspace_tdvp_blas(A, H, **kwargs)
    elif A.backend() == "cuda":
        return _subspace_tdvp_cuda(A, H, **kwargs)
    else:
        raise RuntimeError("Backend not recognised.")


class TDVP(metaclass=ABCMeta):
    """The base class for all Time Dependent Variational Principle implementations."""

    def __new__(
        cls,
        A: Union[ttn, ms_ttn],
        H: Union[sop_operator, ms_sop_operator],
        expansion: str = "onesite",
        **kwargs,
    ) -> "TDVP":
        """A factory method for constructing an object used for performing either single or multi set tdvp calculations.
        Which type to construct is determined by the types of the input A and h matrices.

        :param A: Tree Tensor Network that the tdvp algorithm will act on
        :type A: ttn | ms_ttn
        :param H: The Hamiltonian sop operator object
        :type H: sop_operator | ms_sop_operator
        :param expansion: A string determining the type of bond dimension expansion to be used.  Either no subspace expansion ('onesite') or energy variance based ('subspace').  (Default: 'onesite')
        :type expansion: {'onesite', 'subspace'}, optional
        :param `**kwargs`: Keyword arguments to pass to the tdvp engine constructor. The allowed values depend on the choice of expansion:
    
            - **krylov_dim** (int, optional): The krylov subspace dimension used for the eigensolver steps. (Default: 16)
            - **num_threads** (int, optional): The number of openmp threads to be used by the solver. (Default: 1)
            - **nstep** (int, optional): The number of internal steps to use for evaluation of the matrix exponential. (Default: 1)
            - **subspace_krylov_dim** (int, optional): Only when expansion="subspace". The subspace expansion based krylov subspace dimension. This is only used if expansion="subspace". (Default: 6)
            - **subspace_neigs** (int, optional): Only when expansion="subspace".The number of eigenvalues to evaluate when performing the subspace steps. This is only used if expansion="subspace". (Default: 2)
            - **set_var_num_threads** (int, optional): The number of openmp threads to be used by the solver for parallelising over the set variable. (Default: 1)

        :returns: The tdvp evaluation object
        :rtype: TDVP
        """
        if A.backend() != H.backend():
            raise RuntimeError(
                "Hamiltonian and operator do not have compatible backends."
            )

        if expansion == "onesite":
            return _one_site_tdvp(A, H, **kwargs)
        elif expansion == "subspace":
            return _subspace_tdvp(A, H, **kwargs)
        else:
            raise RuntimeError("Invalid input types for TDVP.")

    @abstractmethod
    def assign(self, o: "TDVP"):
        """Assign the value of the tdvp engine from another

        :param o: The TDVP object to copy into this one
        :type o: TDVP
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the TDVP object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the TDVP object"""
        pass

    @property
    @abstractmethod
    def coefficient(self) -> Union[float, complex]:
        """A coefficient used to scale the timestep."""
        pass

    @property
    @abstractmethod
    def t(self) -> float:
        """The current time point reached by the integrator."""
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        """The timestep used for integration of the dynamics."""
        pass

    @property
    @abstractmethod
    def expmv_tol(self) -> float:
        """The tolerance used for the krylov subspace matrix exponential."""
        pass

    @property
    @abstractmethod
    def krylov_steps(self) -> int:
        """The number of internal substeps used by the krylov subspace integrator for each real step."""
        pass

    @property
    @abstractmethod
    def use_time_dependent_hamiltonian(self) -> bool:
        """Whether or not to update the time variable of the Hamiltonian object throughout integration."""
        pass

    @abstractmethod
    def clear(self):
        """Clear and deallocate all internal buffers of the TDVP object"""
        pass

    @abstractmethod
    def step(self, A: ttn, H: sop_operator, update_env: bool = False):
        """Performs a single step of the single site TDVP algorithm

        :param A: The Tree Tensor Network Object that will be optimised using the TDVP algorithm
        :type A: ttn
        :param H: The Hamiltonian sop operator object
        :type H: sop_operator
        :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
        :type update_env: bool, optional
        """
        pass

    @abstractmethod
    def __call__(self, A: ttn, H: sop_operator, update_env: bool = False):
        """Performs a single step of the single site TDVP algorithm

        :param A: The Tree Tensor Network Object that will be optimised using the TDVP algorithm
        :type A: ttn
        :param H: The Hamiltonian sop operator object
        :type H: sop_operator
        :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
        :type update_env: bool, optional
        """
        pass

    @abstractmethod
    def prepare_environment(
        self, A: ttn, H: sop_operator, attempt_expansion: bool = False
    ):
        """Update all Single Particle Function environment tensors to prepare the system for performing a TDVP sweep.

        :param A: The Tree Tensor Network Object that will be optimised using the TDVP algorithm
        :type A: ttn
        :param H: The Hamiltonian sop operator object
        :type H: sop_operator
        :param attempt_expansion: Whether or not attempt to expand bond dimensions throughout this step.  (Default: False)
        :type attempt_expansion: bool, optional
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the TDVP

        :return: The backend type of the object
        :rtype: str
        """
        pass


class OneSiteTDVP(TDVP):
    """The base class for all one-site tdvp algorithm implementations."""

    def __new__(
        cls,
        A: Union[ttn, ms_ttn],
        H: Union[sop_operator, ms_sop_operator],
        krylov_dim: int = 16,
        num_threads: int = 1,
        set_var_num_threads: int = 1,
        nstep: int = 1
    ) -> "OneSiteTDVP":
        """A factory method for constructing an object used for performing single set tdvp calculations

        :param A: Tree Tensor Network that the tdvp algorithm will act on
        :type A: Union[ttn, ms_ttn]
        :param H: The Hamiltonian sop operator object
        :type H: Union[sop_operator, ms_sop_operator]
        :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
        :type krylov_dim: int, optional
        :param num_threads: The number of openmp threads to be used by the solver for parallelising over the Hamiltonian sum. (Default: 1)
        :type num_threads: int, optional
        :param set_var_num_threads: The number of openmp threads to be used by the solver for parallelising over the set variable. (Default: 1)
        :type set_var_num_threads: int, optional
        :param nstep: The number of internal steps to use for evaluation of the matrix exponential. (Default: 1)
        :type nstep: (int, optional)

        :returns: The tdvp evaluation object
        :rtype: OneSiteTDVP
        """
        if A.backend() != H.backend():
            raise RuntimeError(
                "Hamiltonian and operator do not have compatible backends."
            )

        return _one_site_tdvp(A, H, krylov_dim=krylov_dim, num_threads=num_threads, nstep=nstep, set_var_num_threads=set_var_num_threads)

    def initialise(
        A: Union[ttn, ms_ttn],
        H: Union[sop_operator, ms_sop_operator],
        krylov_dim: int = 16,
        num_threads: int = 1,
        set_var_num_threads: int = 1,
        nstep: int = 1,
    ) -> "OneSiteTDVP":
        """Initialise the internal buffers of the tdvp object needed to perform tdvp on a Tree Tensor Network A, with Hamiltonian H.

        :param A: Tree Tensor Network that the tdvp algorithm will act on
        :type A: Union[ttn, ms_ttn]
        :param H: The Hamiltonian sop operator object
        :type H: Union[sop_operator, ms_sop_operator]
        :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
        :type krylov_dim: int, optional
        :param num_threads: The number of openmp threads to be used by the solver for parallelising over the Hamiltonian sum. (Default: 1)
        :type num_threads: int, optional
        :param nstep: The number of internal steps to use for evaluation of the matrix exponential. (Default: 1)
        :type nstep: (int, optional)

        :returns: The tdvp evaluation object
        :rtype: OneSiteTDVP
        """


OneSiteTDVP.register(one_site_tdvp_complex)
OneSiteTDVP.register(multiset_one_site_tdvp_complex)
if _cuda_import:
    OneSiteTDVP.register(one_site_tdvp_complex_cuda)
    OneSiteTDVP.register(multiset_one_site_tdvp_complex_cuda)


class SubspaceExpansionTDVP(TDVP):
    """The base class for all subspace expansion based tdvp implementations.  This set of implementations
    allows for adaptive growth of the bond dimensions used throughout the tensor network throughout the
    optimisation process.  In order to do this, the code uses two metrics:

    :Two-site Energy Variance:
        This criteria expands the basis based on the use of the variance of the two-site energy.  In particular,
        this finds the dominant contribution to the two-site energy variance that does not lay in the current
        one-site manifold and expands the one-site tensors in these directions, if the weighting of these through
        an update step is larger than some user defined tolerance.

    :One-site Natural Population:
        This criteria expands the basis based on the occupancy of the one-site energy density matrix.  This spawns
        new random basis functions if there are not sufficiently many (a parameter controllued by the user)
        unoccupied basis functions, with the threshold for unoccupied being set by the user.
    """

    def __new__(
        cls,
        A: Union[ttn, ms_ttn],
        H: Union[sop_operator, ms_sop_operator],
        krylov_dim: int = 16,
        subspace_krylov_dim: int = 4,
        subspace_neigs: int = 2,
        num_threads: int = 1,
        nstep: int = 1
    ) -> "SubspaceExpansionTDVP":
        """A factory method for constructing an object used for performing single set tdvp calculations

        :param A: Tree Tensor Network that the tdvp algorithm will act on
        :type A: ttn | ms_ttn
        :param H: The Hamiltonian sop operator object
        :type H: sop_operator | ms_sop_operator
        :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
        :type krylov_dim: int, optional
        :param subspace_krylov_dim: The subspace expansion based krylov subspace dimension. This is only used if expansion="subspace". (Default: 6)
        :type subspace_krylov_dim: int, optional
        :param subspace_neigs: The number of eigenvalues to evaluate when performing the subspace steps. This is only used if expansion="subspace". (Default: 2)
        :type subspace_neigs: int, optional
        :param num_threads: The number of openmp threads to be used by the solver. (Default: 1)
        :type num_threads: int, optional
        :param nstep: The number of internal steps to use for evaluation of the matrix exponential. (Default: 1)
        :type nstep: (int, optional)

        :returns: The tdvp evaluation object
        :rtype: SubspaceExpansionTDVP
        """
        if A.backend() != H.backend():
            raise RuntimeError(
                "Hamiltonian and operator do not have compatible backends."
            )

        return _subspace_tdvp(
            A,
            H,
            krylov_dim=krylov_dim,
            subspace_krylov_dim=subspace_krylov_dim,
            subspace_neigs=subspace_neigs,
            num_threads=num_threads,
            nstep=nstep
        )


    def initialise(
        A: Union[ttn, ms_ttn],
        H: Union[sop_operator, ms_sop_operator],
        krylov_dim: int = 16,
        subspace_krylov_dim: int = 4,
        subspace_neigs: int = 2,
        num_threads: int = 1,
        nstep: int = 1
    ) :
        """Initialise adaptive one-site TDVP object initialising all buffers needed to perform TDVP on a Tree Tensor Network A, with Hamiltonian H.

        :param A: Tree Tensor Network that the tdvp algorithm will act on
        :type A: ttn | ms_ttn
        :param H: The Hamiltonian sop operator object
        :type H: sop_operator | ms_sop_operator
        :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
        :type krylov_dim: int, optional
        :param subspace_krylov_dim: The subspace expansion based krylov subspace dimension. This is only used if expansion="subspace". (Default: 6)
        :type subspace_krylov_dim: int, optional
        :param subspace_neigs: The number of eigenvalues to evaluate when performing the subspace steps. This is only used if expansion="subspace". (Default: 2)
        :type subspace_neigs: int, optional
        :param num_threads: The number of openmp threads to be used by the solver. (Default: 1)
        :type num_threads: int, optional
        :param nstep: The number of internal steps to use for evaluation of the matrix exponential. (Default: 1)
        :type nstep: (int, optional)
        """

    @property
    def subspace_eigensolver_tol(self) -> float:
        """The absolute tolerance of the krylov subspace eigensolver used for subspace expansion"""
        pass

    @property
    def subspace_eigensolver_reltol(self) -> float:
        """The relative tolerance of the krylov subspace eigensolver used for subspace expansion"""
        pass

    @property
    def spawning_threshold(self) -> float:
        """The threshold variable used to determine whether or not to spawn a new 
        basis vector using the two-site energy variance criteria
        """
        pass

    @property
    def unoccupied_threshold(self) -> float:
        """The threshold variable used to determine whether or not to spawn a new 
        basis vector using the spawning critea
        """
        pass

    @property
    def subspace_weighting_factor(self) -> float:
        """A coefficient used to weight the importance of the second order contributions.  Taken as 1 for the tdvp algorithm"""
        pass

    @property
    def only_apply_when_no_unoccupied(self) -> bool:
        """A flag to set whether or not to apply the subspace expansion scheme at all times or only when there are no unoccupied vectors """
        pass

    @property 
    def eval_but_dont_aply(self) -> bool:
        """A flag to set whether to evaluate the metric for subspace expansion but not to apply the 
        results. This should only be used for timing executation of the subspace expansion scheme."""
        pass

    @property
    def minimum_unoccupied(self) -> int:
        """The minimum number of unoccupied variables required at each subspace expansion step.  
        If fewer are detected, additional vectors will be added to reach this limit."""
        pass

    @property
    def maximum_bond_dimension(self) -> int:
        """The maximum bond dimension we can expand to through a subspace expansion step."""
        pass

SubspaceExpansionTDVP.register(multiset_one_site_tdvp_complex)
if _cuda_import:
    SubspaceExpansionTDVP.register(multiset_one_site_tdvp_complex_cuda)

tdvp = TDVP
