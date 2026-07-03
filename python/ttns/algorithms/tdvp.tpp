/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTHON_BINDINGTDVPTPP
#define PYTHON_BINDINGTDVPTPP

#include "tdvp.hpp"
#include "common_bindings.hpp"

namespace py = pybind11;

namespace python_bindings
{
    template<typename Alg>
    void bind_tdvp_fixed_chi(py::class_<Alg>& cls, const std::string& alg_name)
    {
        using _ttn = typename Alg::ttn_type;
        using _sop = typename Alg::env_type;
        using size_type = typename Alg::size_type;
        cls.def(py::init<>());
        cls.def(py::init<const _ttn &, const _sop &, size_type, size_type, size_type, size_type>(), 
            py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("nstep") = 1, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1, 
            docs::constructor(alg_name, false, true).c_str());
        cls.def("initialise", &Alg::initialise, 
            py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("nstep") = 1, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1,  
            docs::initialise(alg_name, false, true).c_str());
        
    }
    template<typename Alg>
    void bind_tdvp_adaptive_chi(py::class_<Alg>& cls, const std::string& alg_name)
    {
        using _ttn = typename Alg::ttn_type;
        using _sop = typename Alg::env_type;
        using size_type = typename Alg::size_type;

        cls.def(py::init<>());
        cls.def(py::init<const _ttn &, const _sop &, size_type, size_type, size_type, size_type, size_type, size_type>(),
             py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("nstep") = 1,  py::arg("subspace_krylov_dim") = 6, py::arg("subspace_neigs") = 2, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1, 
             docs::constructor(alg_name, true, true).c_str());
        cls.def("initialise", &Alg::initialise,
            py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("nstep") = 1, py::arg("subspace_krylov_dim") = 4, py::arg("subspace_neigs") = 2, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1,  
            docs::initialise(alg_name, true, true).c_str());
    }
}


template <typename T, template <typename, typename> class ttn_class, typename backend>
void init_tdvp_onesite(py::module &m, const std::string &label)
{
    using namespace ttns;

    using tdvp = _one_site_tdvp<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename tdvp::env_type;

    using size_type = typename tdvp::size_type;
    using real_type = typename linalg::get_real_type<T>::type;
    // wrapper for the sPOP type

    // utils::eigenvalue_target& mode(){return m_eigensolver.mode();}
    // const utils::eigenvalue_target& mode() const{return m_eigensolver.mode();}
    auto cls = py::class_<tdvp>(m, label.c_str());

    using namespace python_bindings;
    bind_tdvp_fixed_chi(cls, std::string("TDVP"));
    bind_sweeping_common<backend>(cls, std::string("TDVP"));

    BIND_RW_PROPERTY(cls, tdvp, T, coefficient, "A coefficient used to scale the timestep.");
    BIND_RW_PROPERTY(cls, tdvp, real_type, t, "The current time point reached by the integrator.");
    BIND_RW_PROPERTY(cls, tdvp, real_type, dt, "The timestep used for integration of the dynamics.");
    BIND_RW_PROPERTY(cls, tdvp, real_type, expmv_tol, "The tolerance used for the krylov subspace matrix exponential.");
    BIND_RW_PROPERTY(cls, tdvp, size_type, krylov_steps, "The number of internal substeps used by the krylov subspace integrator for each real step.");
    BIND_RW_PROPERTY(cls, tdvp, bool, use_time_dependent_hamiltonian, "Whether or not to update the time variable of the Hamiltonian object throughout integration.");


    bind_dtype<T>(cls);
    bind_copyable(cls);
    bind_pickleable(cls);


}

template <typename T, template <typename, typename> class ttn_class, typename backend>
void init_tdvp_adaptive(py::module &m, const std::string &label)
{
    using namespace ttns;

    using atdvp = _adaptive_one_site_tdvp<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename atdvp::env_type;

    using size_type = typename atdvp::size_type;
    using real_type = typename linalg::get_real_type<T>::type;

    auto cls = py::class_<atdvp>(m, label.c_str());

    using namespace python_bindings;
    bind_tdvp_adaptive_chi(cls, std::string("adaptive TDVP"));
    bind_sweeping_common<backend>(cls, std::string("adaptive TDVP"));

    BIND_RW_PROPERTY(cls, atdvp, T, coefficient, "A coefficient used to scale the timestep.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, t, "The current time point reached by the integrator.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, dt, "The timestep used for integration of the dynamics.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, expmv_tol, "The tolerance used for the krylov subspace matrix exponential.");
    BIND_RW_PROPERTY(cls, atdvp, size_type, krylov_steps, "The number of internal substeps used by the krylov subspace integrator for each real step.");
    BIND_RW_PROPERTY(cls, atdvp, bool, use_time_dependent_hamiltonian, "Whether or not to update the time variable of the Hamiltonian object throughout integration.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, subspace_eigensolver_tol, "The absolute tolerance of the krylov subspace eigensolver used for subspace expansion.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, subspace_eigensolver_reltol, "The relative tolerance of the krylov subspace eigensolver used for subspace expansion.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, spawning_threshold, "The singular value threshold variable used to determine whether or not to spawn a new basis vector.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, unoccupied_threshold, "The variable used to determine whether or not to spawn a new basis vector based on the variables being occupied.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, subspace_weighting_factor,  "A coefficient used to weight the importance of the second order contributions.  Taken as 1 for the DMRG algorithm.");
    BIND_RW_PROPERTY(cls, atdvp, bool, only_apply_when_no_unoccupied,  "A flag to set whether or not to apply the subspace expansion scheme at all times or only when there are no unoccupied vectors.");
    BIND_RW_PROPERTY(cls, atdvp, bool, eval_but_dont_apply, "A flag to set whether to evaluate the metric for subspace expansion but not to apply the results. This should only be used for timing executation of the subspace expansion scheme.");
    BIND_RW_PROPERTY(cls, atdvp, size_type, minimum_unoccupied, "The minimum number of unoccupied variables required at each subspace expansion step.  If fewer are detected, additional vectors will be added to reach this limit.");
    BIND_RW_PROPERTY(cls, atdvp, size_type, maximum_bond_dimension, "The maximum bond dimension we can expand to through a subspace expansion step.");

    bind_dtype<T>(cls);
    bind_copyable(cls);
    bind_pickleable(cls);
}

template <typename T, template <typename, typename> class ttn_class, typename backend>
void init_tdvp_adaptive_ts_cost(py::module &m, const std::string &label)
{
    using namespace ttns;


    using atdvp = _subspace_two_site_cost_tdvp<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename atdvp::env_type;

    using size_type = typename atdvp::size_type;
    using real_type = typename linalg::get_real_type<T>::type;

    auto cls = py::class_<atdvp>(m, label.c_str());

    using namespace python_bindings;
    bind_tdvp_fixed_chi(cls, std::string("twosite TDVP"));
    bind_sweeping_common<backend>(cls, std::string("twosite TDVP"));

    BIND_RW_PROPERTY(cls, atdvp, T, coefficient, "A coefficient used to scale the timestep.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, t, "The current time point reached by the integrator.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, dt, "The timestep used for integration of the dynamics.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, expmv_tol, "The tolerance used for the krylov subspace matrix exponential.");
    BIND_RW_PROPERTY(cls, atdvp, size_type, krylov_steps, "The number of internal substeps used by the krylov subspace integrator for each real step.");
    BIND_RW_PROPERTY(cls, atdvp, bool, use_time_dependent_hamiltonian, "Whether or not to update the time variable of the Hamiltonian object throughout integration.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, subspace_eigensolver_tol, "The absolute tolerance of the krylov subspace eigensolver used for subspace expansion.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, subspace_eigensolver_reltol, "The relative tolerance of the krylov subspace eigensolver used for subspace expansion.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, spawning_threshold, "The singular value threshold variable used to determine whether or not to spawn a new basis vector.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, unoccupied_threshold, "The variable used to determine whether or not to spawn a new basis vector based on the variables being occupied.");
    BIND_RW_PROPERTY(cls, atdvp, real_type, subspace_weighting_factor,  "A coefficient used to weight the importance of the second order contributions.  Taken as 1 for the DMRG algorithm.");
    BIND_RW_PROPERTY(cls, atdvp, bool, only_apply_when_no_unoccupied,  "A flag to set whether or not to apply the subspace expansion scheme at all times or only when there are no unoccupied vectors.");
    BIND_RW_PROPERTY(cls, atdvp, bool, eval_but_dont_apply, "A flag to set whether to evaluate the metric for subspace expansion but not to apply the results. This should only be used for timing executation of the subspace expansion scheme.");
    BIND_RW_PROPERTY(cls, atdvp, size_type, minimum_unoccupied, "The minimum number of unoccupied variables required at each subspace expansion step.  If fewer are detected, additional vectors will be added to reach this limit.");
    BIND_RW_PROPERTY(cls, atdvp, size_type, maximum_bond_dimension, "The maximum bond dimension we can expand to through a subspace expansion step.");

    bind_dtype<T>(cls);
    bind_copyable(cls);
    bind_pickleable(cls);
}

#endif
