#ifndef PYTHON_BINDING_DMRG_HPP
#define PYTHON_BINDING_DMRG_HPP

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


template <typename T, template <typename, typename> class ttn_class>
void init_dmrg_core(py::module& m, const std::string& label)
{
    using namespace ttns;
    using backend = linalg::blas_backend;

    using dmrg = _one_site_dmrg<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename dmrg::env_type;

    using size_type = typename dmrg::size_type;
    using real_type = typename linalg::get_real_type<T>::type;
    //wrapper for the sPOP type 
    py::class_<dmrg>(m, label.c_str())
        .def(py::init<>())
        .def(py::init<const _ttn&, const _sop&, size_type, size_type>(),
                  py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("num_threads")=1)
        .def("assign", [](dmrg& self, const dmrg& o){self=o;})
        .def("__copy__",[](const dmrg& o){return dmrg(o);})
        .def("__deepcopy__", [](const dmrg& o, py::dict){return dmrg(o);}, py::arg("memo"))
        .def("initialise", &dmrg::initialise, py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("num_threads")=1)
        .def("E", &dmrg::E)
        .def_property
            (
                "restarts", 
                static_cast<const size_type& (dmrg::*)() const>(&dmrg::restarts),
                [](dmrg& o, const size_type& i){o.restarts() = i;}
            )
        .def_property
            (
                "eigensolver_tol", 
                static_cast<const real_type& (dmrg::*)() const>(&dmrg::eigensolver_tol),
                [](dmrg& o, const real_type& i){o.eigensolver_tol() = i;}
            )
        .def_property
            (
                "eigensolver_reltol", 
                static_cast<const real_type& (dmrg::*)() const>(&dmrg::eigensolver_reltol),
                [](dmrg& o, const real_type& i){o.eigensolver_reltol() = i;}
            )
        .def("clear", &dmrg::clear)
        .def("step", &dmrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false)
        .def("__call__", &dmrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false)
        .def("prepare_environment", &dmrg::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion")=false);
    //utils::eigenvalue_target& mode(){return m_eigensolver.mode();}
    //const utils::eigenvalue_target& mode() const{return m_eigensolver.mode();}
}

template <typename T, template <typename, typename> class ttn_class>
void init_dmrg_adaptive(py::module& m, const std::string& label)
{
    using namespace ttns;
    using backend = linalg::blas_backend;

    using admrg = _adaptive_one_site_dmrg<T, backend, ttn_class>;
    using _ttn = ttn_class<T, backend>;
    using _sop = typename admrg::env_type;

    using size_type = typename admrg::size_type;
    using real_type = typename linalg::get_real_type<T>::type;

    //wrapper for the sPOP type 
    py::class_<admrg>(m, label.c_str())
        .def(py::init<>())
        .def(py::init<const _ttn&, const _sop&, size_type, size_type, size_type, size_type>(),
                  py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("subspace_krylov_dim")=4, py::arg("subspace_neigs")=2, py::arg("num_threads")=1)
        .def("assign", [](admrg& self, const admrg& o){self=o;})
        .def("__copy__",[](const admrg& o){return admrg(o);})
        .def("__deepcopy__", [](const admrg& o, py::dict){return admrg(o);}, py::arg("memo"))
        .def("initialise", &admrg::initialise, py::arg(), py::arg(), py::arg("krylov_dim")=16, py::arg("subspace_krylov_dim")=4, py::arg("subspace_neigs")=2, py::arg("num_threads")=1)
        .def("E", &admrg::E)
        .def_property
            (
                "restarts", 
                static_cast<const size_type& (admrg::*)() const>(&admrg::restarts),
                [](admrg& o, const size_type& i){o.restarts() = i;}
            )
        .def_property
            (
                "eigensolver_tol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::eigensolver_tol),
                [](admrg& o, const real_type& i){o.eigensolver_tol() = i;}
            )
        .def_property
            (
                "eigensolver_reltol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::eigensolver_reltol),
                [](admrg& o, const real_type& i){o.eigensolver_reltol() = i;}
            )
        .def_property
            (
                "subspace_eigensolver_tol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::subspace_eigensolver_tol),
                [](admrg& o, const real_type& i){o.subspace_eigensolver_tol() = i;}
            )
        .def_property
            (
                "subspace_eigensolver_reltol", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::subspace_eigensolver_reltol),
                [](admrg& o, const real_type& i){o.subspace_eigensolver_reltol() = i;}
            )
        .def_property
            (
                "spawning_threshold", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::spawning_threshold),
                [](admrg& o, const real_type& i){o.spawning_threshold() = i;}
            )
        .def_property
            (
                "unoccupied_threshold", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::unoccupied_threshold),
                [](admrg& o, const real_type& i){o.unoccupied_threshold() = i;}
            )
        .def_property
            (
                "subspace_weighting_factor", 
                static_cast<const real_type& (admrg::*)() const>(&admrg::subspace_weighting_factor),
                [](admrg& o, const real_type& i){o.subspace_weighting_factor() = i;}
            )
        .def_property
            (
                "only_apply_when_no_unoccupied", 
                static_cast<const bool& (admrg::*)() const>(&admrg::only_apply_when_no_unoccupied),
                [](admrg& o, bool i){o.only_apply_when_no_unoccupied() = i;}
            )
        .def_property
            (
                "eval_but_dont_apply", 
                static_cast<const bool& (admrg::*)() const>(&admrg::eval_but_dont_apply),
                [](admrg& o, bool i){o.eval_but_dont_apply() = i;}
            )
        .def_property
            (
                "minimum_unoccupied", 
                static_cast<const size_type& (admrg::*)() const>(&admrg::minimum_unoccupied),
                [](admrg& o, const size_type& i){o.minimum_unoccupied() = i;}
            )
        .def_property
            (
                "maximum_bond_dimension", 
                static_cast<const size_type& (admrg::*)() const>(&admrg::maximum_bond_dimension),
                [](admrg& o, const size_type& i){o.maximum_bond_dimension() = i;}
            )

        .def("neigenvalues", &admrg::neigenvalues)

        .def("clear", &admrg::clear)
        .def("step", &admrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false)
        .def("__call__", &admrg::operator(), py::arg(), py::arg(), py::arg("update_env") = false)
        .def("prepare_environment", &admrg::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion")=false);

    //orthogonality::truncation_mode& truncation_mode() {return m_ss_expand.truncation_mode();}
    //const orthogonality::truncation_mode& truncation_mode() const {return m_ss_expand.truncation_mode();}

}

template <typename T>
void init_dmrg(py::module &m, const std::string& label)
{
    init_dmrg_core<T, ttns::ttn>(m, (std::string("one_site_dmrg_")+label));
    init_dmrg_adaptive<T, ttns::ttn>(m, (std::string("adaptive_one_site_dmrg_")+label));
    init_dmrg_core<T, ttns::ms_ttn>(m, (std::string("multiset_one_site_dmrg_")+label));
}

void initialise_dmrg(py::module& m);



#endif

