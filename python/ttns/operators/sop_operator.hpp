#ifndef PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP
#define PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP

#include <ttns_lib/operators/sop_operator.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T>
void init_sop_operator(py::module &m, const std::string& label)
{
    using namespace ttns;

    using real_type = typename linalg::get_real_type<T>::type;
    using ttn_type = ttn<T, linalg::blas_backend>;
    using opdict = operator_dictionary<T, linalg::blas_backend>;

    using _sop = sop_operator<T, linalg::blas_backend>;
    //the base primitive operator type
    py::class_<_sop>(m, (std::string("sop_operator_")+label).c_str())
        .def(py::init())
        .def(py::init<const _sop&>())
        .def(py::init<SOP<T>&, const ttn_type&, const system_modes&, bool, bool, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def(py::init<SOP<T>&, const ttn_type&, const system_modes&, const opdict&, bool, bool, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("assign", [](_sop& self, const _sop& o){self=o;})
        .def("__copy__",[](const _sop& o){return _sop(o);})
        .def("__deepcopy__", [](const _sop& o, py::dict){return _sop(o);}, py::arg("memo"))
        .def_property
            (
                "Eshift", 
                static_cast<const T& (_sop::*)() const>(&_sop::Eshift),
                [](_sop& o, const T& i){o.Eshift() = i;}
            )
        .def(
              "initialise", 
              [](_sop& o, SOP<T>& sop, const ttn_type& A, const system_modes& sys, bool compress, bool exploit_identity, bool use_sparse)
              {
                  o.initialise(sop, A, sys, compress, exploit_identity, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](_sop& o, SOP<T>& sop, const ttn_type& A, const system_modes& sys, const opdict& opd, bool compress, bool exploit_identity, bool use_sparse)
              {
                  o.initialise(sop, A, sys, opd, compress, exploit_identity, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true
            )
        .def("clear", &_sop::clear)
        .def("update", &_sop::template update<real_type>)
        .def("nterms", &_sop::nterms)
        .def("nmodes", &_sop::nmodes);
}

void initialise_sop_operator(py::module& m);
#endif  //PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP


