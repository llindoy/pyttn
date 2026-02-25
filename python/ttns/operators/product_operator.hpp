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

#ifndef PYTHON_BINDING_TTNS_PRODUCT_OPERATOR_HPP
#define PYTHON_BINDING_TTNS_PRODUCT_OPERATOR_HPP

#include <ttns_lib/operators/product_operator.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include "../../utils.hpp"

namespace py = pybind11;

template <typename T, typename backend>
void init_product_operator(py::module &m, const std::string &label)
{
    using namespace ttns;

    using opdict = operator_dictionary<T, backend>;
    using pop = product_operator<T, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    // the base primitive operator type
    py::class_<pop>(m, (std::string("product_operator_") + label).c_str())
        .def(py::init())
        .def(py::init<const pop &>())
        .def(py::init<sNBO<real_type> &, const system_modes &, bool>(),
             py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sNBO<real_type> &, const system_modes &, const opdict &, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sNBO<T> &, const system_modes &, bool>(),
             py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sNBO<T> &, const system_modes &, const opdict &, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sPOP &, const system_modes &, bool>(),
             py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sPOP &, const system_modes &, const opdict &, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sOP &, const system_modes &, bool>(),
             py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sOP &, const system_modes &, const opdict &, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("assign", [](pop &self, const pop &o)
             { self = o; })
        .def("__copy__", [](const pop &o)
             { return pop(o); })
        .def("__deepcopy__", [](const pop &o, py::dict)
             { return pop(o); }, py::arg("memo"))
        .def("initialise", [](pop &o, sNBO<T> &sop, const system_modes &sys, bool use_sparse)
             { o.initialise(sop, sys, use_sparse); }, py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", [](pop &o, sNBO<T> &sop, const system_modes &sys, const opdict &opd, bool use_sparse)
             { o.initialise(sop, sys, opd, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", [](pop &o, sNBO<real_type> &sop, const system_modes &sys, bool use_sparse)
             { o.initialise(sop, sys, use_sparse); }, py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", [](pop &o, sNBO<real_type> &sop, const system_modes &sys, const opdict &opd, bool use_sparse)
             { o.initialise(sop, sys, opd, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", [](pop &o, sPOP &sop, const system_modes &sys, bool use_sparse)
             { o.initialise(sop, sys, use_sparse); }, py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", [](pop &o, sPOP &sop, const system_modes &sys, const opdict &opd, bool use_sparse)
             { o.initialise(sop, sys, opd, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", [](pop &o, sOP &sop, const system_modes &sys, bool use_sparse)
             { o.initialise(sop, sys, use_sparse); }, py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", [](pop &o, sOP &sop, const system_modes &sys, const opdict &opd, bool use_sparse)
             { o.initialise(sop, sys, opd, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("clear", &pop::clear)
        .def("nmodes", &pop::nmodes)
        .def("complex_dtype", [](const pop &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", [](const pop &o)
             {std::ostringstream oss; oss << o; return oss.str(); })

#ifdef CEREAL_LIBRARY_FOUND
         .def("save", 
            [](const pop & a, const std::string& ofname, bool as_binary){serialisation_utilities::save_obj(a, ofname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
        .def("load", 
            [](pop & a, const std::string& ifname, bool as_binary){serialisation_utilities::load_obj(a, ifname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
         .def(py::pickle(
            [](const pop& a){return serialisation_utilities::__getstate__(a);},
            [](py::tuple t){return serialisation_utilities::__setstate__<pop>(t);}
         ))  
#endif  

        .def("backend", [](const pop &)
             { return linalg::traits<backend>::label(); });
}

template <typename real_type, typename backend>
void initialise_product_operator(py::module &m)
{
    using complex_type = std::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_product_operator<real_type, backend>(m, "real");
#endif
    init_product_operator<complex_type, backend>(m, "complex");
}

#endif
