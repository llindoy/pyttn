#ifndef PYTHON_BINDING_TTNS_OPERATOR_DICTIONARY_HPP
#define PYTHON_BINDING_TTNS_OPERATOR_DICTIONARY_HPP

#include <ttns_lib/operators/sop_operator.hpp>
#include <sstream>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T, typename backend>
void init_operator_dictionary(py::module &m, const std::string& label)
{
    using namespace ttns;

    using opdict = operator_dictionary<T, backend>;
    using dict_type = typename opdict::dict_type;
    using elem_type = typename opdict::elem_type;
    //the base primitive operator type
    py::class_<opdict>(m, (std::string("operator_dictionary_")+label).c_str())
        .def(py::init())
        .def(py::init<size_t>())
        .def(py::init<const dict_type&>())
        .def(py::init<const opdict&>())
        .def("assign", [](opdict& self, const opdict& o){self=o;})
        .def("assign", [](opdict& self, const dict_type& o){self=o;})
        .def("__copy__",[](const opdict& o){return opdict(o);})
        .def("__deepcopy__", [](const opdict& o, py::dict){return opdict(o);}, py::arg("memo"))

        .def("clear", &opdict::clear)
        .def("resize", &opdict::resize)

        .def("__setitem__", [](opdict& o, size_t i, const elem_type& el){o[i] = el;})
        .def("__getitem__", [](opdict& o, size_t i)->elem_type& {return o[i];}, py::return_value_policy::reference)
                
        .def("site_dictionary", [](opdict& o, size_t i){return o.site_dictionary(i);})

        .def("insert", &opdict::insert)
        .def("__call__", [](const opdict& o, size_t nu, const std::string & l){return o(nu, l);})

        .def("__len__", &opdict::size)
        .def("nmodes", &opdict::nmodes)


        .def("__str__", [](const opdict& o)
        {
            std::ostringstream oss;
            for(size_t i = 0; i < o.nmodes(); ++i)
            {
                oss << "mode: " << i << std::endl;
                for(const auto& t : o[i])
                {
                    oss << t.first << " " << t.second.to_string() << std::endl;
                }
            }
            return oss.str();
        })
        .def("backend", [](const opdict&){return backend::label();});

}

template <typename real_type, typename backend>
void initialise_operator_dictionary(py::module& m)
{
    using complex_type = linalg::complex<real_type>;
  
    init_operator_dictionary<real_type, backend>(m, "real");
    init_operator_dictionary<complex_type, backend>(m, "complex");
}
#endif  //PYTHON_BINDING_TTNS_OPERATOR_DICTIONARY_HPP


