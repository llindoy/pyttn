#ifndef PYTHON_BINDING_TDVP_HPP
#define PYTHON_BINDING_TDVP_HPP

#include <ttns_lib/ttn/ttn.hpp>
#include <ttns_lib/ttn/ms_ttn.hpp>
#include <ttns_lib/sweeping_algorithm/tdvp.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>


namespace py=pybind11;


template <typename T, template <typename, typename> class ttn_class>
void init_tdvp_core(py::module& m, const std::string& label)
{
    using namespace ttns;
    using backend = linalg::blas_backend;

    using tdvp = _one_site_tdvp<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename tdvp::env_type;

    using size_type = typename tdvp::size_type;
    using real_type = typename linalg::get_real_type<T>::type;
    //wrapper for the sPOP type 
    py::class_<tdvp>(m, label.c_str())
        .def(py::init<>())
        .def(py::init<const _ttn&, const _sop&, size_type, size_type, size_type>(),
                  py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("nstep")=1, py::arg("num_threads")=1)
        .def("assign", [](tdvp& self, const tdvp& o){self=o;})
        .def("__copy__",[](const tdvp& o){return tdvp(o);})
        .def("__deepcopy__", [](const tdvp& o, py::dict){return tdvp(o);}, py::arg("memo"))
        .def("initialise", &tdvp::initialise, py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("nstep")=1, py::arg("num_threads")=1)
        .def_property
            (
                "coefficient", 
                static_cast<const T& (tdvp::*)() const>(&tdvp::coefficient),
                [](tdvp& o, const T& i){o.coefficient() = i;}
            )
        .def_property
            (
                "t", 
                static_cast<const real_type& (tdvp::*)() const>(&tdvp::t),
                [](tdvp& o, const real_type& i){o.t() = i;}
            )
        .def_property
            (
                "dt", 
                static_cast<const real_type& (tdvp::*)() const>(&tdvp::dt),
                [](tdvp& o, const real_type& i){o.dt() = i;}
            )
        .def_property
            (
                "expmv_tol", 
                static_cast<const real_type& (tdvp::*)() const>(&tdvp::expmv_tol),
                [](tdvp& o, const real_type& i){o.expmv_tol() = i;}
            )
        .def_property
            (
                "krylov_steps", 
                static_cast<const size_type& (tdvp::*)() const>(&tdvp::krylov_steps),
                [](tdvp& o, const size_type& i){o.krylov_steps() = i;}
            )
        .def_property
            (
                "use_time_dependent_hamiltonian",
                static_cast<const bool& (tdvp::*)() const>(&tdvp::use_time_dependent_hamiltonian),
                [](tdvp& o, const bool& i){o.use_time_dependent_hamiltonian() = i;}
            )
        .def("clear", &tdvp::clear)
        .def("step", &tdvp::operator(), py::arg(), py::arg(), py::arg("update_env") = false)
        .def("__call__", &tdvp::operator(), py::arg(), py::arg(), py::arg("update_env") = false)
        .def("prepare_environment", &tdvp::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion")=false);

}


template <typename T, template <typename, typename> class ttn_class>
void init_tdvp_adaptive(py::module& m, const std::string& label)
{
    using namespace ttns;
    using backend = linalg::blas_backend;

    using atdvp = _adaptive_one_site_tdvp<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename atdvp::env_type;

    using size_type = typename atdvp::size_type;
    using real_type = typename linalg::get_real_type<T>::type;

    py::class_<atdvp>(m, label.c_str())
        .def(py::init<>())
        .def(py::init<const _ttn&, const _sop&, size_type, size_type, size_type, size_type, size_type>(),
                  py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("nstep")=16, py::arg("subspace_krylov_dim")=4, py::arg("subspace_neigs")=2, py::arg("num_threads")=1)
        .def("assign", [](atdvp& self, const atdvp& o){self=o;})
        .def("__copy__",[](const atdvp& o){return atdvp(o);})
        .def("__deepcopy__", [](const atdvp& o, py::dict){return atdvp(o);}, py::arg("memo"))
        .def("initialise", &atdvp::initialise, py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("nstep")=16, py::arg("subspace_krylov_dim")=4, py::arg("subspace_neigs")=2, py::arg("num_threads")=1)
        .def_property
            (
                "coefficient", 
                static_cast<const T& (atdvp::*)() const>(&atdvp::coefficient),
                [](atdvp& o, const T& i){o.coefficient() = i;}
            )
        .def_property
            (
                "t", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::t),
                [](atdvp& o, const real_type& i){o.t() = i;}
            )
        .def_property
            (
                "dt", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::dt),
                [](atdvp& o, const real_type& i){o.dt() = i;}
            )
        .def_property
            (
                "expmv_tol", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::expmv_tol),
                [](atdvp& o, const real_type& i){o.expmv_tol() = i;}
            )
        .def_property
            (
                "krylov_steps", 
                static_cast<const size_type& (atdvp::*)() const>(&atdvp::krylov_steps),
                [](atdvp& o, const size_type& i){o.krylov_steps() = i;}
            )
        .def_property
            (
                "use_time_dependent_hamiltonian",
                static_cast<const bool& (atdvp::*)() const>(&atdvp::use_time_dependent_hamiltonian),
                [](atdvp& o, const bool& i){o.use_time_dependent_hamiltonian() = i;}
            )
        .def_property
            (
                "subspace_eigensolver_tol", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::subspace_eigensolver_tol),
                [](atdvp& o, const real_type& i){o.subspace_eigensolver_tol() = i;}
            )
        .def_property
            (
                "subspace_eigensolver_reltol", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::subspace_eigensolver_reltol),
                [](atdvp& o, const real_type& i){o.subspace_eigensolver_reltol() = i;}
            )
        .def_property
            (
                "spawning_threshold", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::spawning_threshold),
                [](atdvp& o, const real_type& i){o.spawning_threshold() = i;}
            )
        .def_property
            (
                "unoccupied_threshold", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::unoccupied_threshold),
                [](atdvp& o, const real_type& i){o.unoccupied_threshold() = i;}
            )
        .def_property
            (
                "subspace_weighting_factor", 
                static_cast<const real_type& (atdvp::*)() const>(&atdvp::subspace_weighting_factor),
                [](atdvp& o, const real_type& i){o.subspace_weighting_factor() = i;}
            )
        .def_property
            (
                "only_apply_when_no_unoccupied", 
                static_cast<const bool& (atdvp::*)() const>(&atdvp::only_apply_when_no_unoccupied),
                [](atdvp& o, bool i){o.only_apply_when_no_unoccupied() = i;}
            )
        .def_property
            (
                "eval_but_dont_apply", 
                static_cast<const bool& (atdvp::*)() const>(&atdvp::eval_but_dont_apply),
                [](atdvp& o, bool i){o.eval_but_dont_apply() = i;}
            )
        .def_property
            (
                "minimum_unoccupied", 
                static_cast<const size_type& (atdvp::*)() const>(&atdvp::minimum_unoccupied),
                [](atdvp& o, const size_type& i){o.minimum_unoccupied() = i;}
            )
        .def_property
            (
                "maximum_bond_dimension", 
                static_cast<const size_type& (atdvp::*)() const>(&atdvp::maximum_bond_dimension),
                [](atdvp& o, const size_type& i){o.maximum_bond_dimension() = i;}
            )
        .def("clear", &atdvp::clear)
        .def("step", [](atdvp& o, _ttn& A, _sop& sop, bool update_environment=false){return o(A, sop, update_environment);}, py::arg(), py::arg(), py::arg("update_env") = false)
        .def("__call__", [](atdvp& o, _ttn& A, _sop& sop, bool update_environment=false){return o(A, sop, update_environment);}, py::arg(), py::arg(), py::arg("update_env") = false)
        .def("prepare_environment", &atdvp::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion")=false);
}

template <typename T>
void init_tdvp(py::module &m, const std::string& label)
{
    init_tdvp_core<T, ttns::ttn>(m, (std::string("one_site_tdvp_")+label));
    init_tdvp_adaptive<T, ttns::ttn>(m, (std::string("adaptive_one_site_tdvp_")+label));
    init_tdvp_core<T, ttns::ms_ttn>(m, (std::string("multiset_one_site_tdvp_")+label));
}

void initialise_tdvp(py::module& m);


#endif

