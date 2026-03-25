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

#ifndef PYTHON_BINDING_TTNS_OPERATOR_DICTIONARY_TPP
#define PYTHON_BINDING_TTNS_OPERATOR_DICTIONARY_TPP

#include "operator_dictionary.hpp"

namespace py = pybind11;

template <typename T, typename backend>
void init_operator_dictionary(py::module &m, const std::string &label)
{
    using namespace ttns;

    using opdict = operator_dictionary<T, backend>;
    using dict_type = typename opdict::dict_type;
    using elem_type = typename opdict::elem_type;
    // the base primitive operator type
    py::class_<opdict>(m, (std::string("operator_dictionary_") + label).c_str())
        .def(py::init())
        .def(py::init<size_t>())
        .def(py::init<const dict_type &>())
        .def(py::init<const opdict &>())
        .def("assign", [](opdict &self, const opdict &o)
             { self = o; })
        .def("assign", [](opdict &self, const dict_type &o)
             { self = o; })
        .def("__copy__", [](const opdict &o)
             { return opdict(o); })
        .def("__deepcopy__", [](const opdict &o, py::dict)
             { return opdict(o); }, py::arg("memo"))

        .def("clear", &opdict::clear)
        .def("resize", &opdict::resize)

        .def("__setitem__", [](opdict &o, size_t i, const elem_type &el)
             { o[i] = el; })
        .def("__getitem__", [](opdict &o, size_t i) -> elem_type &
             { return o[i]; }, py::return_value_policy::reference)

        .def("site_dictionary", [](opdict &o, size_t i)
             { return o.site_dictionary(i); })

        .def("insert", &opdict::insert)
        .def("__call__", [](const opdict &o, size_t nu, const std::string &l)
             { return o(nu, l); })

        .def("__len__", &opdict::size)
        .def("nmodes", &opdict::nmodes)

        .def("__str__", [](const opdict &o)
             {
            std::ostringstream oss;
            for(size_t i = 0; i < o.nmodes(); ++i)
            {
                oss << "mode: " << i << std::endl;
                for(const auto& t : o[i])
                {
                    oss << std::get<0>(t) << " " << std::get<1>(t).to_string() << std::endl;
                }
            }
            return oss.str(); })
        .def("backend", [](const opdict &)
             { return linalg::traits<backend>::label(); })
#ifdef CEREAL_LIBRARY_FOUND
         .def("save", 
            [](const opdict & a, const std::string& ofname, bool as_binary){serialisation_utilities::save_obj(a, ofname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
        .def("load", 
            [](opdict & a, const std::string& ifname, bool as_binary){serialisation_utilities::load_obj(a, ifname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
         .def(py::pickle(
            [](const opdict& a){return serialisation_utilities::__getstate__(a);},
            [](py::tuple t){return serialisation_utilities::__setstate__<opdict>(t);}
         ))
#endif
             
             ;
}

template <typename real_type, typename backend>
void initialise_operator_dictionary_types(py::module &m)
{
    using complex_type = std::complex<real_type>;

    init_operator_dictionary<real_type, backend>(m, "real");
    init_operator_dictionary<complex_type, backend>(m, "complex");
}
#endif // PYTHON_BINDING_TTNS_OPERATOR_DICTIONARY_TPP
