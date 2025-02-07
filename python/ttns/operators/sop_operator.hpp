#ifndef PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP
#define PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP

#include <ttns_lib/sop/system_information.hpp>
#include <ttns_lib/operators/sop_operator.hpp>
#include <ttns_lib/operators/multiset_sop_operator.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T, typename backend>
void init_sop_operator(py::module &m, const std::string& label)
{
    using namespace ttns;

    using real_type = typename linalg::get_real_type<T>::type;
    using ttn_type = ttn<T, backend>;
    using opdict = operator_dictionary<T, backend>;
    using _sop = sop_operator<T, backend>;

    using ms_ttn_type = ms_ttn<T, backend>;
    using _mssop = multiset_sop_operator<T, backend>;

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

    py::class_<_mssop>(m, (std::string("multiset_sop_operator_")+label).c_str())
        .def(py::init())
        .def(py::init<const _mssop&>())
        .def(py::init<multiset_SOP<T>&, const ms_ttn_type&, const system_modes&, bool, bool, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def(py::init<multiset_SOP<T>&, const ms_ttn_type&, const system_modes&, const opdict&, bool, bool, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("assign", [](_mssop& self, const _mssop& o){self=o;})
        .def("__copy__",[](const _mssop& o){return _mssop(o);})
        .def("__deepcopy__", [](const _mssop& o, py::dict){return _mssop(o);}, py::arg("memo"))
        .def(
              "initialise", 
              [](_mssop& o, multiset_SOP<T>& sop, const ms_ttn_type& A, const system_modes& sys, bool compress, bool exploit_identity, bool use_sparse)
              {
                  o.initialise(sop, A, sys, compress, exploit_identity, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](_mssop& o, multiset_SOP<T>& sop, const ms_ttn_type& A, const system_modes& sys, const opdict& opd, bool compress, bool exploit_identity, bool use_sparse)
              {
                  o.initialise(sop, A, sys, opd, compress, exploit_identity, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress")=true, py::arg("identity_opt") = true, py::arg("use_sparse") = true
            )
        .def("clear", &_mssop::clear)
        .def("update", &_mssop::template update<real_type>)
        .def("nset", &_mssop::nset)
        .def("nmodes", &_mssop::nmodes);
}

template <typename backend>
void initialise_sop_operator(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_sop_operator<real_type, backend>(m, "real");
#endif
    init_sop_operator<complex_type, backend>(m, "complex");
}
#endif  //PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP


