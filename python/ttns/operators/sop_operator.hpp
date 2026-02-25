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

#ifndef PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP
#define PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP

#include <ttns_lib/sop/system_information.hpp>
#include <ttns_lib/operators/sop_operator.hpp>
#include <ttns_lib/operators/multiset_sop_operator.hpp>
#include "../../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template <typename T, typename backend>
void init_sop_operator(py::module &m, const std::string &label)
{
    using namespace ttns;

    using real_type = typename linalg::get_real_type<T>::type;
    using ttn_type = ttn<T, backend>;
    using opdict = operator_dictionary<T, backend>;
    using _sop = sop_operator<T, backend>;
    using coef = literal::coeff<T>;

    using ms_ttn_type = ms_ttn<T, backend>;
    using _mssop = multiset_sop_operator<T, backend>;

    // the base primitive operator type
    py::class_<_sop>(m, (std::string("sop_operator_") + label).c_str())
        .def(py::init())
        .def(py::init<const _sop &>())
        .def(py::init<SOP<T> &, const ttn_type &, const system_modes &, bool, bool, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def(py::init<SOP<T> &, const ttn_type &, const system_modes &, const opdict &, bool, bool, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("assign", [](_sop &self, const _sop &o)
             { self = o; })
        .def("__copy__", [](const _sop &o)
             { return _sop(o); })
        .def("__deepcopy__", [](const _sop &o, py::dict)
             { return _sop(o); }, py::arg("memo"))
        .def_property("Eshift", static_cast<const T &(_sop::*)() const>(&_sop::Eshift), [](_sop &o, const T &i)
                      { o.set_Eshift(i); })
        .def("set_Eshift", [](_sop& o, const T& i){o.set_Eshift(i);})
        .def("set_Eshift", [](_sop& o, const coef& i){o.set_Eshift(i);})

        .def("initialise", [](_sop &o, SOP<T> &sop, const ttn_type &A, const system_modes &sys, bool compress, bool exploit_identity, bool use_sparse)
             { o.initialise(sop, A, sys, compress, exploit_identity, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("initialise", [](_sop &o, SOP<T> &sop, const ttn_type &A, const system_modes &sys, const opdict &opd, bool compress, bool exploit_identity, bool use_sparse)
             { o.initialise(sop, A, sys, opd, compress, exploit_identity, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("clear", &_sop::clear)
        .def("update", &_sop::template update<real_type>)
        .def("nterms", &_sop::nterms)
        .def("nmodes", &_sop::nmodes)
        .def("complex_dtype", [](const _sop &)
             { return !std::is_same<T, real_type>::value; })
        .def("backend", [](const _sop &)
             { return linalg::traits<backend>::label(); })
#ifdef CEREAL_LIBRARY_FOUND
         .def("save", 
            [](const _sop & a, const std::string& ofname, bool as_binary){serialisation_utilities::save_obj(a, ofname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
        .def("load", 
            [](_sop & a, const std::string& ifname, bool as_binary){serialisation_utilities::load_obj(a, ifname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
         .def(py::pickle(
            [](const _sop& a){return serialisation_utilities::__getstate__(a);},
            [](py::tuple t){return serialisation_utilities::__setstate__<_sop>(t);}
         ))  
#endif    
             ;

    py::class_<_mssop>(m, (std::string("multiset_sop_operator_") + label).c_str())
        .def(py::init())
        .def(py::init<const _mssop &>())
        .def(py::init<multiset_SOP<T> &, const ms_ttn_type &, const system_modes &, bool, bool, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def(py::init<multiset_SOP<T> &, const ms_ttn_type &, const system_modes &, const opdict &, bool, bool, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("assign", [](_mssop &self, const _mssop &o)
             { self = o; })
        .def("__copy__", [](const _mssop &o)
             { return _mssop(o); })
        .def("__deepcopy__", [](const _mssop &o, py::dict)
             { return _mssop(o); }, py::arg("memo"))
        .def("Eshift", static_cast<const T &(_mssop::*)(size_t, size_t) const>(&_mssop::Eshift))
        .def("set_Eshift", [](_mssop& o, size_t i, size_t j, const T& a){o.set_Eshift(i, j, a);})
        .def("set_Eshift", [](_mssop& o, size_t i, size_t j, const coef& a){o.set_Eshift(i, j, a);})
        .def("initialise", [](_mssop &o, multiset_SOP<T> &sop, const ms_ttn_type &A, const system_modes &sys, bool compress, bool exploit_identity, bool use_sparse)
             { o.initialise(sop, A, sys, compress, exploit_identity, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("initialise", [](_mssop &o, multiset_SOP<T> &sop, const ms_ttn_type &A, const system_modes &sys, const opdict &opd, bool compress, bool exploit_identity, bool use_sparse)
             { o.initialise(sop, A, sys, opd, compress, exploit_identity, use_sparse); }, py::arg(), py::arg(), py::arg(), py::arg(), py::arg("compress") = true, py::arg("identity_opt") = true, py::arg("use_sparse") = true)
        .def("clear", &_mssop::clear)
        .def("update", &_mssop::template update<real_type>)
        .def("nset", &_mssop::nset)
        .def("complex_dtype", [](const _mssop &)
             { return !std::is_same<T, real_type>::value; })
        .def("nmodes", &_mssop::nmodes)
        .def("backend", [](const _mssop &)
             { return linalg::traits<backend>::label(); })
#ifdef CEREAL_LIBRARY_FOUND
         .def("save", 
            [](const _mssop & a, const std::string& ofname, bool as_binary){serialisation_utilities::save_obj(a, ofname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
        .def("load", 
            [](_mssop & a, const std::string& ifname, bool as_binary){serialisation_utilities::load_obj(a, ifname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
         .def(py::pickle(
            [](const _mssop& a){return serialisation_utilities::__getstate__(a);},
            [](py::tuple t){return serialisation_utilities::__setstate__<_mssop>(t);}
         ))  
#endif  
             
             ;
}

template <typename real_type, typename backend>
void initialise_sop_operator(py::module &m)
{
    using complex_type = std::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_sop_operator<real_type, backend>(m, "real");
#endif
    init_sop_operator<complex_type, backend>(m, "complex");
}
#endif // PYTHON_BINDING_TTNS_SOP_OPERATOR_HPP
