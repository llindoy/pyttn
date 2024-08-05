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


template <typename T>
void init_dmrg(py::module &m, const std::string& label)
{
    init_dmrg_core<T, ttns::ttn>(m, (std::string("one_site_dmrg_")+label));
    init_dmrg_core<T, ttns::ms_ttn>(m, (std::string("multiset_one_site_dmrg_")+label));
}

void initialise_dmrg(py::module& m);



#endif

