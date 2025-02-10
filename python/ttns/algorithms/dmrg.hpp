#ifndef PYTHON_BINDING_DMRG_HPP
#define PYTHON_BINDING_DMRG_HPP

#include "../../utils.hpp"

#include <ttns_lib/ttn/ttn.hpp>
#include <ttns_lib/ttn/ms_ttn.hpp>
#include <ttns_lib/sweeping_algorithm/dmrg.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>


namespace py=pybind11;


template <typename T, template <typename, typename> class ttn_class, typename backend>
void init_dmrg_core(py::module& m, const std::string& label)
{
    using namespace ttns;

    using _T = typename linalg::numpy_converter<T>::type;

    using dmrg = _one_site_dmrg<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename dmrg::env_type;

    using size_type = typename dmrg::size_type;
    using real_type = typename linalg::get_real_type<T>::type;
    //wrapper for the sPOP type 
    py::class_<dmrg>(m, label.c_str())
        .def(py::init<>())
        .def(py::init<const _ttn&, const _sop&, size_type, size_type>(),
                  py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("num_threads")=1, R"mydelim(
            Construct a new one-site DMRG object initialising all buffers needed to perform DMRG on a Tree Tensor Network A, with Hamiltonian H.

            :Parameters:    - **A** (:class:`ttn_complex`) - The Tree Tensor Network Object that will be optimised using the DMRG algorithm
                            - **H** (:class:`sop_operator_complex`) - The Hamiltonian sop operator object
                            - **krylov_dim** (int, optional) - The krylov subspace dimension used for the eigensolver steps. (Default: 16)
                            - **num_threads** (int, optional) - The number of openmp threads to be used by the solver. (Default: 1)
          )mydelim")
        .def("assign", [](dmrg& self, const dmrg& o){self=o;}, R"mydelim(
            Assign the DMRG object from another DMRG object
          )mydelim")
        .def("__copy__",[](const dmrg& o){return dmrg(o);})
        .def("__deepcopy__", [](const dmrg& o, py::dict){return dmrg(o);}, py::arg("memo"))
        .def("initialise", &dmrg::initialise, py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("num_threads")=1, R"mydelim(
            Initialise the internal buffers of the DMRG object needed to perform DMRG on a Tree Tensor Network A, with Hamiltonian H.

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
            :type krylov_dim: int, optional
            :param numthreads: The number of openmp threads to be used by the solver. (Default: 1)
            :type numthreads: int, optional
          )mydelim")
        .def("E", [](const dmrg& o){return _T(o.E());}, "Returns the current energy computed through the last DMRG sweep.")
        .def_property
            (
                "restarts", 
                static_cast<const size_type& (dmrg::*)() const>(&dmrg::restarts),
                [](dmrg& o, const size_type& i){o.restarts() = i;},
                "The number of restarts to use in the krylov subspace eigensolver."
            )
        .def_property
            (
                "eigensolver_tol", 
                static_cast<const real_type& (dmrg::*)() const>(&dmrg::eigensolver_tol),
                [](dmrg& o, const real_type& i){o.eigensolver_tol() = i;},
                "The absolute tolerance of the krylov subspace eigensolver"
            )
        .def_property
            (
                "eigensolver_reltol", 
                static_cast<const real_type& (dmrg::*)() const>(&dmrg::eigensolver_reltol),
                [](dmrg& o, const real_type& i){o.eigensolver_reltol() = i;},
                "The relative tolerance of the krylov subspace eigensolver"
            )
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
        .def("prepare_environment", &dmrg::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion")=false, R"mydelim(
            Update all Single Particle Function environment tensors to prepare the system for performing a DMRG sweep. 

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
            :type update_env: bool, optional
          )mydelim")
        .def("backend", [](const dmrg&){return backend::label();})
        .doc() = R"mydelim(
            A class implementing the one site DMRG algorithm on trees.
          )mydelim";
    //utils::eigenvalue_target& mode(){return m_eigensolver.mode();}
    //const utils::eigenvalue_target& mode() const{return m_eigensolver.mode();}
}

template <typename T, template <typename, typename> class ttn_class, typename backend>
void init_dmrg_adaptive(py::module& m, const std::string& label)
{
    using namespace ttns;

    using _T = typename linalg::numpy_converter<T>::type;

    using admrg = _adaptive_one_site_dmrg<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename admrg::env_type;

    using size_type = typename admrg::size_type;
    using real_type = typename linalg::get_real_type<T>::type;

    //wrapper for the sPOP type 
    py::class_<admrg>(m, label.c_str())
        .def(py::init<>(), R"mydelim(
            Default construct for adaptive one-site dmrg object.
            )mydelim")
        .def(py::init<const _ttn&, const _sop&, size_type, size_type, size_type, size_type>(),
                  py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("subspace_krylov_dim")=6, py::arg("subspace_neigs")=2, py::arg("num_threads")=1,  R"mydelim(
            Construct a new adaptive one-site DMRG object initialising all buffers needed to perform DMRG on a Tree Tensor Network A, with Hamiltonian H.

            :Parameters:    - **A** (:class:`ttn_complex`) - The Tree Tensor Network Object that will be optimised using the DMRG algorithm
                            - **H** (:class:`sop_operator_complex`) - The Hamiltonian sop operator object
                            - **krylov_dim** (int, optional) - The krylov subspace dimension used for the eigensolver steps. (Default: 16)
                            - **subspace_krylov_dim** The subspace expansion based krylov subspace dimension. This is only used if expansion="subspace". (Default: 6)
                            - **subspace_neigs** (int, optional) - The number of eigenvalues to evaluate when performing the subspace steps. This is only used if expansion="subspace". (Default: 2)
                            - **num_threads** (int, optional) - The number of openmp threads to be used by the solver. (Default: 1)
            )mydelim")
        .def("assign", [](admrg& self, const admrg& o){self=o;})
        .def("__copy__",[](const admrg& o){return admrg(o);})
        .def("__deepcopy__", [](const admrg& o, py::dict){return admrg(o);}, py::arg("memo"))
        .def("initialise", &admrg::initialise, py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("subspace_krylov_dim")=4, py::arg("subspace_neigs")=2, py::arg("num_threads")=1, R"mydelim(
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
            :param numthreads: The number of openmp threads to be used by the solver. (Default: 1)
            :type numthreads: int, optional
            )mydelim")
        .def("E", [](const admrg& o){return _T(o.E());}, "Returns the current energy computed through the last DMRG sweep.")
        .def_property
            (
                "restarts", 
                static_cast<const size_type& (admrg::*)() const>(&admrg::restarts),
                [](admrg& o, const size_type& i){o.restarts() = i;},
                "The number of restarts to use in the krylov subspace eigensolver."
            )
        .def_property
            (
                "eigensolver_tol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::eigensolver_tol),
                [](admrg& o, const real_type& i){o.eigensolver_tol() = i;},
                "The absolute tolerance of the krylov subspace eigensolver"
            )
        .def_property
            (
                "eigensolver_reltol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::eigensolver_reltol),
                [](admrg& o, const real_type& i){o.eigensolver_reltol() = i;},
                "The relative tolerance of the krylov subspace eigensolver"
            )
        .def_property
            (
                "subspace_eigensolver_tol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::subspace_eigensolver_tol),
                [](admrg& o, const real_type& i){o.subspace_eigensolver_tol() = i;},
                "The absolute tolerance of the krylov subspace eigensolver used for subspace expansion"
            )
        .def_property
            (
                "subspace_eigensolver_reltol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::subspace_eigensolver_reltol),
                [](admrg& o, const real_type& i){o.subspace_eigensolver_reltol() = i;},
                "The relative tolerance of the krylov subspace eigensolver used for subspace expansion"
            )
        .def_property
            (
                "spawning_threshold", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::spawning_threshold),
                [](admrg& o, const real_type& i){o.spawning_threshold() = i;},
                "The singular value threshold variable used to determine whether or not to spawn a new basis vector"
            )
        .def_property
            (
                "unoccupied_threshold", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::unoccupied_threshold),
                [](admrg& o, const real_type& i){o.unoccupied_threshold() = i;},
                "The variable used to determine whether or not to spawn a new basis vector based on the variables being occupied"
            )
        .def_property
            (
                "subspace_weighting_factor", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::subspace_weighting_factor),
                [](admrg& o, const real_type& i){o.subspace_weighting_factor() = i;},
                "A coefficient used to weight the importance of the second order contributions.  Taken as 1 for the DMRG algorithm"
            )
        .def_property
            (
                "only_apply_when_no_unoccupied", 
                static_cast<const bool& (admrg::*)() const>(&admrg::only_apply_when_no_unoccupied),
                [](admrg& o, bool i){o.only_apply_when_no_unoccupied() = i;},
                "A flag to set whether or not to apply the subspace expansion scheme at all times or only when there are no unoccupied vectors"
            )
        .def_property
            (
                "eval_but_dont_apply", 
                static_cast<const bool& (admrg::*)() const>(&admrg::eval_but_dont_apply),
                [](admrg& o, bool i){o.eval_but_dont_apply() = i;},
                "A flag to set whether to evaluate the metric for subspace expansion but not to apply the results. This should only be used for timing executation of the subspace expansion scheme."
            )
        .def_property
            (
                "minimum_unoccupied", 
                static_cast<const size_type& (admrg::*)() const>(&admrg::minimum_unoccupied),
                [](admrg& o, const size_type& i){o.minimum_unoccupied() = i;},
                "The minimum number of unoccupied variables required at each subspace expansion step.  If fewer are detected, additional vectors will be added to reach this limit."
            )
        .def_property
            (
                "maximum_bond_dimension", 
                static_cast<const size_type& (admrg::*)() const>(&admrg::maximum_bond_dimension),
                [](admrg& o, const size_type& i){o.maximum_bond_dimension() = i;},
                "The maximum bond dimension we can expand to through a subspace expansion step."
            )
        .def("neigenvalues", &admrg::neigenvalues, "Returns the number of eigenvalues that will be evaluated through the subspace expansion step." )

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
        .def("prepare_environment", &admrg::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion")=false, R"mydelim(
            Update all Single Particle Function environment tensors to prepare the system for performing a DMRG sweep. 

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param attempt_expansion: Whether or not to attempt subspace expansion throughout the update scheme.  (Default: False)
            :type attempt_expansion: bool, optional
          )mydelim")
        .def("backend", [](const admrg&){return backend::label();})
        .doc() = R"mydelim(
              A class implementing the adaptive one site DMRG algorithm on trees.
            )mydelim";



    //orthogonality::truncation_mode& truncation_mode() {return m_ss_expand.truncation_mode();}
    //const orthogonality::truncation_mode& truncation_mode() const {return m_ss_expand.truncation_mode();}

}

template <typename T, typename backend>
void init_dmrg(py::module &m, const std::string& label)
{
    init_dmrg_core<T, ttns::ttn, backend>(m, (std::string("one_site_dmrg_")+label));
    init_dmrg_adaptive<T, ttns::ttn, backend>(m, (std::string("adaptive_one_site_dmrg_")+label));
    init_dmrg_core<T, ttns::ms_ttn, backend>(m, (std::string("multiset_one_site_dmrg_")+label));
}

template <typename real_type, typename backend>
void initialise_dmrg(py::module& m)
{
    using complex_type = linalg::complex<real_type>;
  
#ifdef BUILD_REAL_TTN
    //init_dmrg<real_type, backend>(m, "real");
#endif
    init_dmrg<complex_type, backend>(m, "complex");
}


#endif

