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

#ifndef PYTHON_BINDING_DMRGTPP
#define PYTHON_BINDING_DMRGTPP

#include "dmrg.hpp"
#include "../../common_bindings.hpp"

namespace py = pybind11;

template <typename T, template <typename, typename> class ttn_class, typename backend>
void init_dmrg_onesite(py::module &m, const std::string &label)
{
    using namespace ttns;

    using dmrg = _one_site_dmrg<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename dmrg::env_type;

    using size_type = typename dmrg::size_type;
    using real_type = typename linalg::get_real_type<T>::type;
    // wrapper for the sPOP type
    auto cls = py::class_<dmrg>(m, label.c_str());
    cls.def(py::init<>())
        .def(py::init<const _ttn &, const _sop &, size_type, size_type, size_type>(),
             py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1, R"mydelim(
            Construct a new one-site DMRG object initialising all buffers needed to perform DMRG on a Tree Tensor Network A, with Hamiltonian H.

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
            :type krylov_dim: int, optional
            :param num_threads: The number of openmp threads to be used for parallelising over the Hamiltonian sum in the solver. (Default: 1)
            :type num_threads: int, optional
            :param set_var_num_threads: The number of openmp threads to be used for parallelising over the set by the solver. (Default: 1)
            :type set_var_num_threads: int, optional
          )mydelim")
        .def("initialise", &dmrg::initialise, py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1, R"mydelim(
            Initialise the internal buffers of the DMRG object needed to perform DMRG on a Tree Tensor Network A, with Hamiltonian H.

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
            :type krylov_dim: int, optional
            :param num_threads: The number of openmp threads to be used for parallelising over the Hamiltonian sum in the solver. (Default: 1)
            :type num_threads: int, optional
            :param set_var_num_threads: The number of openmp threads to be used for parallelising over the set by the solver. (Default: 1)
            :type set_var_num_threads: int, optional
          )mydelim")
        .def("E", [](const dmrg &o)
             { return T(o.E()); }, "Returns the current energy computed through the last DMRG sweep.")
        .def_property("restarts", static_cast<const size_type &(dmrg::*)() const>(&dmrg::restarts), [](dmrg &o, const size_type &i)
                      { o.restarts() = i; }, "The number of restarts to use in the krylov subspace eigensolver.")
        .def_property("eigensolver_tol", static_cast<const real_type &(dmrg::*)() const>(&dmrg::eigensolver_tol), [](dmrg &o, const real_type &i)
                      { o.eigensolver_tol() = i; }, "The absolute tolerance of the krylov subspace eigensolver")
        .def_property("eigensolver_reltol", static_cast<const real_type &(dmrg::*)() const>(&dmrg::eigensolver_reltol), [](dmrg &o, const real_type &i)
                      { o.eigensolver_reltol() = i; }, "The relative tolerance of the krylov subspace eigensolver")
        .def("clear", &dmrg::clear, "Clear all internal buffers of the DMRG object.")
        .def("step", &dmrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false, R"mydelim(
            Performs a single step of the single site DMRG algorithm

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
            :type update_env: bool, optional
          )mydelim")
        .def("__call__", &dmrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false)
        .def("prepare_environment", &dmrg::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion") = false, R"mydelim(
            Update all Single Particle Function environment tensors to prepare the system for performing a DMRG sweep. 

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
            :type update_env: bool, optional
          )mydelim")
        .def("backend", [](const dmrg &)
             { return linalg::traits<backend>::label(); })


        .doc() = R"mydelim(
            A class implementing the one site DMRG algorithm on trees.
          )mydelim";
    using namespace python_bindings;
    bind_dtype<T>(cls);
    bind_copyable(cls);
    bind_pickleable(cls);
    // utils::eigenvalue_target& mode(){return m_eigensolver.mode();}
    // const utils::eigenvalue_target& mode() const{return m_eigensolver.mode();}
}

template <typename T, template <typename, typename> class ttn_class, typename backend>
void init_dmrg_adaptive(py::module &m, const std::string &label)
{
    using namespace ttns;

    using admrg = _adaptive_one_site_dmrg<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename admrg::env_type;

    using size_type = typename admrg::size_type;
    using real_type = typename linalg::get_real_type<T>::type;

    // wrapper for the sPOP type
    auto cls = py::class_<admrg>(m, label.c_str());
    cls.def(py::init<>(), R"mydelim(
            Default construct for adaptive one-site dmrg object.
            )mydelim")
        .def(py::init<const _ttn &, const _sop &, size_type, size_type, size_type, size_type, size_type>(),
             py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("subspace_krylov_dim") = 6, py::arg("subspace_neigs") = 2, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1, R"mydelim(
            Construct a new adaptive one-site DMRG object initialising all buffers needed to perform DMRG on a Tree Tensor Network A, with Hamiltonian H.

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
            :type krylov_dim: int, optional
            :param subspace_krylov_dim: The subspace expansion based krylov subspace dimension. This is only used if expansion="subspace". (Default: 6)
            :type subspace_krylov_dim: int, optional
            :param subspace_neigs: The number of eigenvalues to evaluate when performing the subspace steps. This is only used if expansion="subspace". (Default: 2)
            :type subspace_neigs: int, optional
            :param num_threads: The number of openmp threads to be used for parallelising over the Hamiltonian sum in the solver. (Default: 1)
            :type num_threads: int, optional
            :param set_var_num_threads: The number of openmp threads to be used for parallelising over the set by the solver. (Default: 1)
            :type set_var_num_threads: int, optional
            )mydelim")
        .def("initialise", &admrg::initialise, py::arg(), py::arg(), py::arg("krylov_dim") = 16, py::arg("subspace_krylov_dim") = 4, py::arg("subspace_neigs") = 2, py::arg("num_threads") = 1, py::arg("set_var_num_threads") = 1, R"mydelim(
            Initialise the internal buffers of the DMRG object needed to perform DMRG on a Tree Tensor Network A, with Hamiltonian H.

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
            :type krylov_dim: int, optional
            :param subspace_krylov_dim: The subspace expansion based krylov subspace dimension. This is only used if expansion="subspace". (Default: 6)
            :type subspace_krylov_dim: int, optional
            :param subspace_neigs: The number of eigenvalues to evaluate when performing the subspace steps. This is only used if expansion="subspace". (Default: 2)
            :type subspace_neigs: int, optional
            :param num_threads: The number of openmp threads to be used for parallelising over the Hamiltonian sum in the solver. (Default: 1)
            :type num_threads: int, optional
            :param set_var_num_threads: The number of openmp threads to be used for parallelising over the set by the solver. (Default: 1)
            :type set_var_num_threads: int, optional
            )mydelim")
        .def("E", [](const admrg &o)
             { return T(o.E()); }, "Returns the current energy computed through the last DMRG sweep.")
        .def_property("restarts", static_cast<const size_type &(admrg::*)() const>(&admrg::restarts), [](admrg &o, const size_type &i)
                      { o.restarts() = i; }, "The number of restarts to use in the krylov subspace eigensolver.")
        .def_property("eigensolver_tol", static_cast<const real_type &(admrg::*)() const>(&admrg::eigensolver_tol), [](admrg &o, const real_type &i)
                      { o.eigensolver_tol() = i; }, "The absolute tolerance of the krylov subspace eigensolver")
        .def_property("eigensolver_reltol", static_cast<const real_type &(admrg::*)() const>(&admrg::eigensolver_reltol), [](admrg &o, const real_type &i)
                      { o.eigensolver_reltol() = i; }, "The relative tolerance of the krylov subspace eigensolver")
        .def_property("subspace_eigensolver_tol", static_cast<const real_type &(admrg::*)() const>(&admrg::subspace_eigensolver_tol), [](admrg &o, const real_type &i)
                      { o.subspace_eigensolver_tol() = i; }, "The absolute tolerance of the krylov subspace eigensolver used for subspace expansion")
        .def_property("subspace_eigensolver_reltol", static_cast<const real_type &(admrg::*)() const>(&admrg::subspace_eigensolver_reltol), [](admrg &o, const real_type &i)
                      { o.subspace_eigensolver_reltol() = i; }, "The relative tolerance of the krylov subspace eigensolver used for subspace expansion")
        .def_property("spawning_threshold", static_cast<const real_type &(admrg::*)() const>(&admrg::spawning_threshold), [](admrg &o, const real_type &i)
                      { o.spawning_threshold() = i; }, "The singular value threshold variable used to determine whether or not to spawn a new basis vector")
        .def_property("unoccupied_threshold", static_cast<const real_type &(admrg::*)() const>(&admrg::unoccupied_threshold), [](admrg &o, const real_type &i)
                      { o.unoccupied_threshold() = i; }, "The variable used to determine whether or not to spawn a new basis vector based on the variables being occupied")
        .def_property("subspace_weighting_factor", static_cast<const real_type &(admrg::*)() const>(&admrg::subspace_weighting_factor), [](admrg &o, const real_type &i)
                      { o.subspace_weighting_factor() = i; }, "A coefficient used to weight the importance of the second order contributions.  Taken as 1 for the DMRG algorithm")
        .def_property("only_apply_when_no_unoccupied", static_cast<const bool &(admrg::*)() const>(&admrg::only_apply_when_no_unoccupied), [](admrg &o, bool i)
                      { o.only_apply_when_no_unoccupied() = i; }, "A flag to set whether or not to apply the subspace expansion scheme at all times or only when there are no unoccupied vectors")
        .def_property("eval_but_dont_apply", static_cast<const bool &(admrg::*)() const>(&admrg::eval_but_dont_apply), [](admrg &o, bool i)
                      { o.eval_but_dont_apply() = i; }, "A flag to set whether to evaluate the metric for subspace expansion but not to apply the results. This should only be used for timing executation of the subspace expansion scheme.")
        .def_property("minimum_unoccupied", static_cast<const size_type &(admrg::*)() const>(&admrg::minimum_unoccupied), [](admrg &o, const size_type &i)
                      { o.minimum_unoccupied() = i; }, "The minimum number of unoccupied variables required at each subspace expansion step.  If fewer are detected, additional vectors will be added to reach this limit.")
        .def_property("maximum_bond_dimension", static_cast<const size_type &(admrg::*)() const>(&admrg::maximum_bond_dimension), [](admrg &o, const size_type &i)
                      { o.maximum_bond_dimension() = i; }, "The maximum bond dimension we can expand to through a subspace expansion step.")
        .def("neigenvalues", &admrg::neigenvalues, "Returns the number of eigenvalues that will be evaluated through the subspace expansion step.")

        .def("clear", &admrg::clear, "Clear all internal buffers of the DMRG object.")
        .def("step", &admrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false, R"mydelim(
            Performs a single step of the single site DMRG algorithm

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
            :type update_env: bool, optional
          )mydelim")

        .def("__call__", &admrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false, R"mydelim(
            Performs a single step of the single site DMRG algorithm

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
            :type update_env: bool, optional
          )mydelim")
        .def("prepare_environment", &admrg::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion") = false, R"mydelim(
            Update all Single Particle Function environment tensors to prepare the system for performing a DMRG sweep. 

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param attempt_expansion: Whether or not to attempt subspace expansion throughout the update scheme.  (Default: False)
            :type attempt_expansion: bool, optional
          )mydelim")
        .def("backend", [](const admrg &)
             { return linalg::traits<backend>::label(); })
        .doc() = R"mydelim(
              A class implementing the adaptive one site DMRG algorithm on trees.
            )mydelim";
    using namespace python_bindings;
    bind_dtype<T>(cls);
    bind_copyable(cls);
    bind_pickleable(cls);
    // orthogonality::truncation_mode& truncation_mode() {return m_ss_expand.truncation_mode();}
    // const orthogonality::truncation_mode& truncation_mode() const {return m_ss_expand.truncation_mode();}
}

#endif
