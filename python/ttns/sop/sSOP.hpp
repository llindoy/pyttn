#ifndef PYTHON_BINDING_STRING_SOP_HPP
#define PYTHON_BINDING_STRING_SOP_HPP

#include <ttns_lib/sop/coeff_type.hpp>
#include <ttns_lib/sop/sSOP.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>


namespace py=pybind11;

template <typename real_type>
void init_sSOP(py::module& m)
{
    using complex_type = linalg::complex<real_type>;
    using namespace ttns;
    using namespace literal;
    //wrapper for the sOP type 
    py::class_<sOP>(m, "sOP")
        .def(py::init())
        .def(py::init<const std::string&, size_t>())
        .def(py::init<const std::string&, size_t, bool>())
        .def(py::init<const sOP&>())
        .def("clear", &sOP::clear)
        .def("assign", [](sOP& self, const sOP& o){self=o;})
        .def("__copy__",[](const sOP& o){return sOP(o);})
        .def("__deepcopy__", [](const sOP& o, py::dict){return sOP(o);}, py::arg("memo"))
        .def_property
            (
                "op", 
                static_cast<const std::string& (sOP::*)() const>(&sOP::op),
                [](sOP& o, const std::string& i){o.op() = i;}
            )
        .def_property
            (
                "mode", 
                static_cast<const size_t& (sOP::*)() const>(&sOP::mode),
                [](sOP& o, const size_t& i){o.mode() = i;}
            )

        .def("__str__", [](const sOP& o){return static_cast<std::string>(o);})

        .def("__add__", [](const sOP& a, const sOP& b){return a+b;})
        .def("__add__", [](const sOP& a, const sPOP& b){return a+b;})
        .def("__add__", [](const sOP& a, const sNBO<real_type>& b){return a+b;})
        .def("__add__", [](const sOP& a, const sNBO<complex_type>& b){return a+b;})
        .def("__add__", [](const sOP& a, const sSOP<real_type>&  b){return a+b;})
        .def("__add__", [](const sOP& a, const sSOP<complex_type>& b){return a+b;})

        .def("__sub__", [](const sOP& a, const sOP& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sPOP& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sNBO<real_type>& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sNBO<complex_type>& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sSOP<real_type>& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sSOP<complex_type>& b){return a-b;})

        .def("__div__", [](const sOP& a, const real_type& b){return a*(1.0/b);})
        .def("__div__", [](const sOP& a, const complex_type& b){return a*(1.0/b);})

        .def("__mul__", [](const sOP& a, const coeff<real_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const real_type& b){return a*b;})
        .def("__mul__", [](const sOP& a, const complex_type& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sOP& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sPOP& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sNBO<real_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sNBO<complex_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sSOP<real_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sSOP<complex_type>& b){return a*b;})

        .def("__rmul__", [](const sOP& a, const coeff<real_type>& b){return a*b;})
        .def("__rmul__", [](const sOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__rmul__", [](const sOP& a, const real_type& b){return a*b;})
        .def("__rmul__", [](const sOP& a, const complex_type& b){return a*b;});



    m.def("fermion_operator", &fermion_operator);
    m.def("fOP", &fermion_operator);

    //wrapper for the sPOP type 
    py::class_<sPOP>(m, "sPOP")
        .def(py::init())
        .def(py::init<const sOP&>())
        .def(py::init<const std::list<sOP>&>())
        .def(py::init<const sPOP&>())
        .def("assign", [](sPOP& self, const sPOP& o){self=o;})
        .def("__copy__",[](const sPOP& o){return sPOP(o);})
        .def("__deepcopy__", [](const sPOP& o, py::dict){return sPOP(o);}, py::arg("memo"))
        .def("clear", &sPOP::clear)
        .def("insert_front", &sPOP::append)
        .def("insert_back", &sPOP::prepend)
        .def("size", &sPOP::size)
        .def("nmodes", &sPOP::nmodes)
        .def_property
            (
                "ops", 
                static_cast<const std::list<sOP>& (sPOP::*)() const>(&sPOP::ops),
                [](sPOP& o, const std::list<sOP>& i){o.ops() = i;}
            )

        .def(
                "__iter__",
                [](sPOP& s){return py::make_iterator(s.begin(), s.end());},
                py::keep_alive<0, 1>()
            )
        .def("__str__", [](const sPOP& o){return static_cast<std::string>(o);})
        .def("__imul__", [](sPOP& a, const sPOP& b){return a*=b;})
        .def("__imul__", [](sPOP& a, const sOP& b){return a*=b;})

        .def("__add__", [](const sPOP& a, const sOP& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sPOP& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sNBO<real_type>& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sNBO<complex_type>& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sSOP<real_type>&  b){return a+b;})
        .def("__add__", [](const sPOP& a, const sSOP<complex_type>& b){return a+b;})

        .def("__sub__", [](const sPOP& a, const sOP& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sPOP& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sNBO<real_type>& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sNBO<complex_type>& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sSOP<real_type>& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sSOP<complex_type>& b){return a-b;})

        .def("__div__", [](const sOP& a, const real_type& b){return a*(1.0/b);})
        .def("__div__", [](const sOP& a, const complex_type& b){return a*(1.0/b);})

        .def("__mul__", [](const sPOP& a, const coeff<real_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const real_type& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const complex_type& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sOP& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sPOP& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sNBO<real_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sNBO<complex_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sSOP<real_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sSOP<complex_type>& b){return a*b;})

        .def("__rmul__", [](const sPOP& a, const coeff<real_type>& b){return a*b;})
        .def("__rmul__", [](const sPOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__rmul__", [](const sPOP& a, const real_type& b){return a*b;})
        .def("__rmul__", [](const sPOP& a, const complex_type& b){return a*b;});

    {
        using coef = coeff<real_type>;
        //using complex_func_type = std::function<complex_type(real_type)>;
        using func_type = std::function<real_type(real_type)>;
        py::class_<coef>(m, "coeff_real")
            .def(py::init())
            .def(py::init<const real_type&>())
            .def(py::init<const func_type&>())
            .def(py::init<const coef&>())
            .def("assign", [](coef& self, const coef& o){self=o;})
            .def("assign", [](coef& self, const real_type& o){self=o;})
            .def("assign", [](coef& self, const func_type& o){self=o;})
            .def("__copy__",[](const coef& o){return coef(o);})
            .def("__deepcopy__", [](const coef& o, py::dict){return coef(o);}, py::arg("memo"))
            .def("clear", &coef::clear)
            .def("is_zero", &coef::is_zero, py::arg("tol")=real_type(1e-14))
            .def("is_positive", &coef::is_positive)
            .def("is_time_dependent", &coef::is_time_dependent)
            .def("__call__", &coef::operator())

            .def("__str__", [](const coef& o){std::ostringstream oss; oss << o; return oss.str();})
            .def("__iadd__", [](coef& a, const real_type& b){return a+=b;})
            .def("__isub__", [](coef& a, const real_type& b){return a-=b;})
            .def("__imul__", [](coef& a, const real_type& b){return a*=b;})
            .def("__idiv__", [](coef& a, const real_type& b){return a/=b;})

            .def("__iadd__", [](coef& a, const coef& b){return a+=b;})
            .def("__isub__", [](coef& a, const coef& b){return a-=b;})
            .def("__imul__", [](coef& a, const coef& b){return a*=b;})

            .def("__div__", [](const coef& a, const real_type& b){return a/b;})
            .def("__div__", [](const coef& a, const complex_type& b){return a/b;})

            .def("__add__", [](coef& a, const coef& b){return a+b;})
            .def("__sub__", [](coef& a, const coef& b){return a-b;})
            .def("__mul__", [](coef& a, const coef& b){return a*b;})
            .def("__add__", [](coef& a, const coeff<complex_type>& b){return a+b;})
            .def("__sub__", [](coef& a, const coeff<complex_type>& b){return a-b;})
            .def("__mul__", [](coef& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sOP& b){return a*b;})
            .def("__mul__", [](coef& a, const sPOP& b){return a*b;})
            .def("__mul__", [](coef& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const coef& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const coef& a, const complex_type& b){return a*b;});
    }
      
    {
        using coef = coeff<complex_type>;
        using func_type = std::function<complex_type(real_type)>;
        using real_func_type = std::function<real_type(real_type)>;
        py::class_<coef>(m, "coeff_complex")
            .def(py::init())
            .def(py::init<const real_type&>())
            .def(py::init<const complex_type&>())
            .def(py::init<const func_type&>())
            .def(py::init<const real_func_type&>())
            .def(py::init<const coef&>())
            .def(py::init<const coeff<real_type>&>())
            .def("assign", [](coef& self, const coef& o){self=o;})
            .def("assign", [](coef& self, const coeff<real_type>& o){self=o;})
            .def("assign", [](coef& self, const real_type& o){self=o;})
            .def("assign", [](coef& self, const complex_type& o){self=o;})
            .def("assign", [](coef& self, const func_type& o){self=o;})
            .def("assign", [](coef& self, const real_func_type& o){self=coeff<real_type>(o);})
            .def("__copy__",[](const coef& o){return coef(o);})
            .def("__deepcopy__", [](const coef& o, py::dict){return coef(o);}, py::arg("memo"))
            .def("clear", &coef::clear)
            .def("is_zero", &coef::is_zero, py::arg("tol")=real_type(1e-14))
            .def("is_positive", &coef::is_positive)
            .def("is_time_dependent", &coef::is_time_dependent)
            .def("__call__", &coef::operator())

            .def("__str__", [](const coef& o){std::ostringstream oss; oss << o; return oss.str();})
            .def("__iadd__", [](coef& a, const real_type& b){return a+=b;})
            .def("__isub__", [](coef& a, const real_type& b){return a-=b;})
            .def("__imul__", [](coef& a, const real_type& b){return a*=b;})
            .def("__idiv__", [](coef& a, const real_type& b){return a/=b;})

            .def("__iadd__", [](coef& a, const complex_type& b){return a+=b;})
            .def("__isub__", [](coef& a, const complex_type& b){return a-=b;})
            .def("__imul__", [](coef& a, const complex_type& b){return a*=b;})
            .def("__idiv__", [](coef& a, const complex_type& b){return a/=b;})

            .def("__iadd__", [](coef& a, const coef& b){return a+=b;})
            .def("__isub__", [](coef& a, const coef& b){return a-=b;})
            .def("__imul__", [](coef& a, const coef& b){return a*=b;})

            .def("__iadd__", [](coef& a, const coeff<real_type>& b){return a+=b;})
            .def("__isub__", [](coef& a, const coeff<real_type>& b){return a-=b;})
            .def("__imul__", [](coef& a, const coeff<real_type>& b){return a*=b;})

            .def("__div__", [](const coef& a, const real_type& b){return a/b;})
            .def("__div__", [](const coef& a, const complex_type& b){return a/b;})

            .def("__add__", [](const coef& a, const coef& b){return a+b;})
            .def("__sub__", [](const coef& a, const coef& b){return a-b;})
            .def("__mul__", [](const coef& a, const coef& b){return a*b;})
            .def("__add__", [](const coef& a, const coeff<real_type>& b){return a+b;})
            .def("__sub__", [](const coef& a, const coeff<real_type>& b){return a-b;})
            .def("__mul__", [](const coef& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sOP& b){return a*b;})
            .def("__mul__", [](const coef& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const coef& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const coef& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const coef& a, const complex_type& b){return a*b;});
    }

    {
        using NBO = sNBO<real_type>;
        //wrapper for the sPOP type 
        py::class_<NBO>(m, "sNBO_real")
            .def(py::init())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const real_type&, const sPOP&>())
            .def(py::init<const real_type&, const sOP&>())
            .def(py::init<const coeff<real_type>&, const sPOP&>())
            .def(py::init<const coeff<real_type>&, const sOP&>())
            .def(py::init<const NBO&>())
            .def("assign", [](NBO& self, const NBO& o){self=o;})
            .def("__copy__",[](const NBO& o){return NBO(o);})
            .def("__deepcopy__", [](const NBO& o, py::dict){return NBO(o);}, py::arg("memo"))
            .def("clear", &NBO::clear)
            .def("insert_front", &NBO::append)
            .def("insert_back", &NBO::prepend)
            .def("nmodes", &NBO::nmodes)
            .def(
                    "__iter__",
                    [](NBO& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "coeff", 
                    static_cast<const coeff<real_type>& (NBO::*)() const>(&NBO::coeff),
                    [](NBO& o, const coeff<real_type>& i){o.coeff() = i;}
                )
            .def_property
                (
                    "ops", 
                    static_cast<const std::list<sOP>& (NBO::*)() const>(&NBO::ops),
                    [](NBO& o, const std::list<sOP>& i){o.ops() = i;}
                )
            .def_property
                (
                    "pop", 
                    static_cast<const sPOP& (NBO::*)() const>(&NBO::pop),
                    [](NBO& o, const sPOP& i){o.pop() = i;}
                )

            .def("__str__", [](const NBO& o){return static_cast<std::string>(o);})
            .def("__imul__", [](NBO& a, const sOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const NBO& b){return a*=b;})
            .def("__imul__", [](NBO& a, const real_type& b){return a*=b;})

            .def("__add__", [](const NBO& a, const sOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sPOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<real_type>&  b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<complex_type>& b){return a+b;})

            .def("__sub__", [](const NBO& a, const sOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<complex_type>& b){return a-b;})

            .def("__div__", [](const NBO& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const NBO& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const complex_type& b){return a*b;});
    }
    {
        using NBO = sNBO<complex_type>;
        //wrapper for the sPOP type 
        py::class_<NBO>(m, "sNBO_complex")
            .def(py::init())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const complex_type&, const sPOP&>())
            .def(py::init<const complex_type&, const sOP&>())
            .def(py::init<const coeff<complex_type>&, const sPOP&>())
            .def(py::init<const coeff<complex_type>&, const sOP&>())
            .def(py::init<const NBO&>())
            .def(py::init<const sNBO<real_type>&>())
            .def("assign", [](NBO& self, const NBO& o){self=o;})
            .def("assign", [](NBO& self, const sNBO<real_type>& o){self=o;})
            .def("__copy__",[](const NBO& o){return NBO(o);})
            .def("__deepcopy__", [](const NBO& o, py::dict){return NBO(o);}, py::arg("memo"))
            .def("clear", &NBO::clear)
            .def("insert_front", &NBO::append)
            .def("insert_back", &NBO::prepend)
            .def("nmodes", &NBO::nmodes)
            .def(
                    "__iter__",
                    [](NBO& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "coeff", 
                    static_cast<const coeff<complex_type>& (NBO::*)() const>(&NBO::coeff),
                    [](NBO& o, const coeff<complex_type>& i){o.coeff() = i;}
                )
            .def_property
                (
                    "ops", 
                    static_cast<const std::list<sOP>& (NBO::*)() const>(&NBO::ops),
                    [](NBO& o, const std::list<sOP>& i){o.ops() = i;}
                )
            .def_property
                (
                    "pop", 
                    static_cast<const sPOP& (NBO::*)() const>(&NBO::pop),
                    [](NBO& o, const sPOP& i){o.pop() = i;}
                )

            .def("__str__", [](const NBO& o){return static_cast<std::string>(o);})
            .def("__imul__", [](NBO& a, const sOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const NBO& b){return a*=b;})
            .def("__imul__", [](NBO& a, const complex_type& b){return a*=b;})

            .def("__add__", [](const NBO& a, const sOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sPOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<real_type>&  b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<complex_type>& b){return a+b;})

            .def("__sub__", [](const NBO& a, const sOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<complex_type>& b){return a-b;})

            .def("__div__", [](const NBO& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const NBO& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const complex_type& b){return a*b;});
    }
    {
        using _SOP = sSOP<real_type>;
        using container_type = typename _SOP::container_type;
        //wrapper for the sPOP type 
        py::class_<_SOP>(m, "sSOP_real")
            .def(py::init())
            .def(py::init<size_t>())
            .def(py::init<const std::string&>())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const sNBO<real_type>&>())
            .def(py::init<const _SOP&>())
            .def("assign", [](_SOP& self, const _SOP& o){self=o;})
            .def("__copy__",[](const _SOP& o){return _SOP(o);})
            .def("__deepcopy__", [](const _SOP& o, py::dict){return _SOP(o);}, py::arg("memo"))
            .def("clear", &_SOP::clear)
            .def("reserve", &_SOP::reserve)
            .def("nmodes", &_SOP::nmodes)
            .def("nterms", &_SOP::nterms)
            .def("__len__",&_SOP::nterms)
            .def(
                    "__iter__",
                    [](_SOP& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "label", 
                    static_cast<const std::string& (_SOP::*)() const>(&_SOP::label),
                    [](_SOP& o, const std::string& i){o.label() = i;}
                )
            .def_property
                (
                    "terms", 
                    static_cast<const container_type& (_SOP::*)() const>(&_SOP::terms),
                    [](_SOP& o, const container_type& i){o.terms() = i;}
                )
            .def(
                    "__setitem__", 
                    [](_SOP& self, size_t i, const sNBO<real_type>& v){self[i] = v;}
                )
            .def(
                    "__getitem__", 
                    static_cast<const sNBO<real_type>& (_SOP::*)(size_t) const>(&_SOP::operator[])
                )
            .def("__str__", [](const _SOP& o){std::ostringstream oss; oss << o; return oss.str();})

            .def("__iadd__", [](_SOP& a, const sOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sPOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sNBO<real_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const _SOP& b){return a+=b;})

            .def("__imul__", [](_SOP& a, const real_type& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sNBO<real_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const _SOP& b){return a*=b;})

            .def("__add__", [](const _SOP& a, const sOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sPOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sSOP<real_type>&  b){return a+b;})
            .def("__add__", [](const _SOP& a, const sSOP<complex_type>& b){return a+b;})

            .def("__sub__", [](const _SOP& a, const sOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sSOP<complex_type>& b){return a-b;})

            .def("__div__", [](const _SOP& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const _SOP& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const complex_type& b){return a*b;});
    }
    {
        using _SOP = sSOP<complex_type>;
        using container_type = typename _SOP::container_type;
        //wrapper for the sPOP type 
        py::class_<_SOP>(m, "sSOP_complex")
            .def(py::init())
            .def(py::init<size_t>())
            .def(py::init<const std::string&>())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const sNBO<real_type>&>())
            .def(py::init<const sNBO<complex_type>&>())
            .def(py::init<const sSOP<real_type>&>())
            .def(py::init<const _SOP&>())
            .def("assign", [](_SOP& self, const sSOP<real_type>& o){self=o;})
            .def("assign", [](_SOP& self, const _SOP& o){self=o;})
            .def("__copy__",[](const _SOP& o){return _SOP(o);})
            .def("__deepcopy__", [](const _SOP& o, py::dict){return _SOP(o);}, py::arg("memo"))
            .def("clear", &_SOP::clear)
            .def("reserve", &_SOP::reserve)
            .def("nmodes", &_SOP::nmodes)
            .def("nterms", &_SOP::nterms)
            .def("__len__",&_SOP::nterms)
            .def(
                    "__iter__",
                    [](_SOP& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "label", 
                    static_cast<const std::string& (_SOP::*)() const>(&_SOP::label),
                    [](_SOP& o, const std::string& i){o.label() = i;}
                )
            .def_property
                (
                    "terms", 
                    static_cast<const container_type& (_SOP::*)() const>(&_SOP::terms),
                    [](_SOP& o, const container_type& i){o.terms() = i;}
                )
            .def(
                    "__setitem__", 
                    [](_SOP& self, size_t i, const sNBO<complex_type>& v){self[i] = v;}
                )
            .def(
                    "__getitem__", 
                    static_cast<const sNBO<complex_type>& (_SOP::*)(size_t) const>(&_SOP::operator[])
                )
            .def("__str__", [](const _SOP& o){std::ostringstream oss; oss << o; return oss.str();})

            .def("__iadd__", [](_SOP& a, const sOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sPOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sNBO<real_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sNBO<complex_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sSOP<real_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const _SOP& b){return a+=b;})

            .def("__imul__", [](_SOP& a, const real_type& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const complex_type& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sNBO<real_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sNBO<complex_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sSOP<real_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const _SOP& b){return a*=b;})

            .def("__add__", [](const _SOP& a, const sOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sPOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sSOP<real_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const _SOP& b){return a+b;})

            .def("__sub__", [](const _SOP& a, const sOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const _SOP& b){return a-b;})

            .def("__div__", [](const _SOP& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const _SOP& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const _SOP& b){return a*b;})

            .def("__rmul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const complex_type& b){return a*b;});
    }
}

void initialise_sSOP(py::module& m);
#endif
