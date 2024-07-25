#ifndef PYTHON_BINDING_MAPPED_SOP_HPP
#define PYTHON_BINDING_MAPPED_SOP_HPP

#include <ttns_lib/sop/sSOP.hpp>
#include <ttns_lib/sop/SOP.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>


namespace py=pybind11;

template <typename T> 
void init_SOP(py::module &m, const std::string& label)
{
    using namespace ttns;

    using real_type = typename linalg::get_real_type<T>::type;
    using _SOP = SOP<T>;
    //wrapper for the sPOP type 
    py::class_<_SOP>(m, label.c_str())
        .def(py::init<size_t>())
        .def(py::init<size_t, const std::string&>())
        .def(py::init<const _SOP&>())
        .def("assign", [](_SOP& self, const _SOP& o){self=o;})
        .def("__copy__",[](const _SOP& o){return _SOP(o);})
        .def("__deepcopy__", [](const _SOP& o, py::dict){return _SOP(o);}, py::arg("memo"))
        .def("clear", &_SOP::clear)
        .def("resize", &_SOP::resize)
        .def("reserve", &_SOP::reserve)
        .def("nmodes", &_SOP::nmodes)
        .def("nterms", &_SOP::nterms)
        .def_property(
            "operator_dictionary", 
            &_SOP::operator_dictionary,
            &_SOP::set_operator_dictionary
         )
        .def("set_operator_dictionary", &_SOP::set_operator_dictionary)
        .def("get_operator_dictionary", &_SOP::operator_dictionary)

        .def("insert", static_cast<void (_SOP::*)(const T&, const sPOP&)>(&_SOP::insert))
        .def("insert", static_cast<void (_SOP::*)(const sNBO<T>&)>(&_SOP::insert))

        .def("set_is_fermion_mode", &_SOP::set_is_fermionic_mode)
        .def("prune_zeros", &_SOP::prune_zeros, py::arg("tol")=1e-15)
        .def("jordan_wigner", static_cast<_SOP& (_SOP::*)(double)>(&_SOP::jordan_wigner), py::arg("tol")=1e-15)
        .def("jordan_wigner", static_cast<_SOP& (_SOP::*)(const system_modes&, double)>(&_SOP::jordan_wigner), py::arg(), py::arg("tol")=1e-15)

        .def_property
            (
                "label", 
                static_cast<const std::string& (_SOP::*)() const>(&_SOP::label),
                [](_SOP& o, const std::string& i){o.label() = i;}
            )
        .def("__str__", [](const _SOP& o){std::ostringstream oss; oss << o; return oss.str();})

        .def("__iadd__", [](_SOP& a, const sOP& b){return a+=b;})
        .def("__iadd__", [](_SOP& a, const sPOP& b){return a+=b;})
        .def("__iadd__", [](_SOP& a, const sNBO<real_type>& b){return a+=b;})
        .def("__iadd__", [](_SOP& a, const sNBO<T>& b){return a+=b;})
        .def("__iadd__", [](_SOP& a, const sSOP<real_type>& b){return a+=b;})
        .def("__iadd__", [](_SOP& a, const sSOP<T>& b){return a+=b;})

        .def("__isub__", [](_SOP& a, const sOP& b){return a-=b;})
        .def("__isub__", [](_SOP& a, const sPOP& b){return a-=b;})
        .def("__isub__", [](_SOP& a, const sNBO<real_type>& b){return a-=b;})
        .def("__isub__", [](_SOP& a, const sNBO<T>& b){return a-=b;})
        .def("__isub__", [](_SOP& a, const sSOP<real_type>& b){return a-=b;})
        .def("__isub__", [](_SOP& a, const sSOP<T>& b){return a-=b;})

        .def("__add__", [](_SOP& a, const sOP& b){return a+b;})
        .def("__add__", [](_SOP& a, const sPOP& b){return a+b;})
        .def("__add__", [](_SOP& a, const sNBO<real_type>& b){return a+b;})
        .def("__add__", [](_SOP& a, const sNBO<T>& b){return a+b;})
        .def("__add__", [](_SOP& a, const sSOP<real_type>& b){return a+b;})
        .def("__add__", [](_SOP& a, const sSOP<T>& b){return a+b;})
        .def("__radd__", [](_SOP& b, const sOP& a){return a+b;})
        .def("__radd__", [](_SOP& b, const sPOP& a){return a+b;})
        .def("__radd__", [](_SOP& b, const sNBO<real_type>& a){return a+b;})
        .def("__radd__", [](_SOP& b, const sNBO<T>& a){return a+b;})
        .def("__radd__", [](_SOP& b, const sSOP<real_type>& a){return a+b;})
        .def("__radd__", [](_SOP& b, const sSOP<T>& a){return a+b;})

        .def("__sub__", [](_SOP& a, const sOP& b){return a-b;})
        .def("__sub__", [](_SOP& a, const sPOP& b){return a-b;})
        .def("__sub__", [](_SOP& a, const sNBO<real_type>& b){return a-b;})
        .def("__sub__", [](_SOP& a, const sNBO<T>& b){return a-b;})
        .def("__sub__", [](_SOP& a, const sSOP<real_type>& b){return a-b;})
        .def("__sub__", [](_SOP& a, const sSOP<T>& b){return a-b;})
        .def("__rsub__", [](_SOP& b, const sOP& a){return a-b;})
        .def("__rsub__", [](_SOP& b, const sPOP& a){return a-b;})
        .def("__rsub__", [](_SOP& b, const sNBO<real_type>& a){return a-b;})
        .def("__rsub__", [](_SOP& b, const sNBO<T>& a){return a-b;})
        .def("__rsub__", [](_SOP& b, const sSOP<real_type>& a){return a-b;})
        .def("__rsub__", [](_SOP& b, const sSOP<T>& a){return a-b;});
}

void initialise_SOP(py::module& m);

#endif
