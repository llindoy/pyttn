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

#ifndef PYTHON_BINDING_MAPPED_STATE_HPP
#define PYTHON_BINDING_MAPPED_STATE_HPP

#include <ttns_lib/sop/state.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include "../../utils.hpp"

namespace py = pybind11;

template <typename T>
void init_state_str(py::module &m)
{
     using value_type = T;
     using numpy_value_type = typename linalg::numpy_converter<T>::type;
     using real_type = typename linalg::get_real_type<T>::type;
     using stateStr = ttns::stateStr;
     // wrapper for the state type
     py::class_<stateStr>(m, "stateStr")
         .def(py::init())
         .def(py::init<const std::vector<size_t> &>())
         .def(py::init<const stateStr &>())
         .def("assign", [](stateStr &self, const stateStr &o)
              { self = o; })
         .def("clear", &stateStr::clear, "For details see :meth:`pyttn.sepState_dtype.clear`")
         .def("resize", &stateStr::resize, "For details see :meth:`pyttn.sepState_dtype.resize`")
         .def("size", &stateStr::size, "For details see :meth:`pyttn.sepState_dtype.size`")
         .def("nnz", &stateStr::nnz, "For details see :meth:`pyttn.sepState_dtype.nnz`")

         .def("state", &stateStr::state, "For details see :meth:`pyttn.sepState_dtype.state`")
         .def("__str__", [](const stateStr &o)
              {std::ostringstream oss; oss << o; return oss.str(); })
         .def("__eq__", [](const stateStr &a, const stateStr &b)
              { return a == b; })
         .def("__ne__", [](const stateStr &a, const stateStr &b)
              { return !(a == b); })

         .def("__mul__", [](const stateStr &a, const real_type &b)
              { return a * value_type(b); })
         .def("__mul__", [](const stateStr &a, const numpy_value_type &b)
              { return a * value_type(b); })
         .def("__rmul__", [](const stateStr &a, const real_type &b)
              { return a * value_type(b); })
         .def("__rmul__", [](const stateStr &a, const numpy_value_type &b)
              { return a * value_type(b); })

         .def("__truediv__", [](const stateStr &a, const real_type &b)
              { return a / value_type(b); })
         .def("__truediv__", [](const stateStr &a, const numpy_value_type &b)
              { return a / value_type(b); })

         .def("__add__", [](stateStr &a, stateStr &b)
              { return add<T>(a, b); })
         .def("__add__", [](stateStr &a, ttns::sepState<T> &b)
              { return a + b; })
         .def("__add__", [](stateStr &a, ttns::sepState<real_type> &b)
              { return a + b; })
         .def("__add__", [](stateStr &a, ttns::ket<T> &b)
              { return a + b; })
         .def("__add__", [](stateStr &a, ttns::ket<real_type> &b)
              { return a + b; })

         .def("__sub__", [](stateStr &a, stateStr &b)
              { return sub<T>(a, b); })
         .def("__sub__", [](stateStr &a, ttns::sepState<T> &b)
              { return a - b; })
         .def("__sub__", [](stateStr &a, ttns::sepState<real_type> &b)
              { return a - b; })
         .def("__sub__", [](stateStr &a, ttns::ket<T> &b)
              { return a - b; })
         .def("__sub__", [](stateStr &a, ttns::ket<real_type> &b)
              { return a - b; });
}

template <typename T>
void init_state(py::module &m, const std::string &label)
{

     using value_type = T;
     using numpy_value_type = typename linalg::numpy_converter<T>::type;
     using real_type = typename linalg::get_real_type<T>::type;
     using complex_type = linalg::complex<real_type>;
     using numpy_complex_type = typename linalg::numpy_converter<complex_type>::type;

     using _sepState = ttns::sepState<T>;
     using stateStr = ttns::stateStr;

     // wrapper for the state type
     py::class_<_sepState>(m, ("sepState_" + label).c_str())
         .def(py::init())
         .def(py::init<const std::vector<size_t> &>())
         .def(py::init<const numpy_value_type &, const std::vector<size_t> &>())
         .def(py::init<const stateStr &>())
         .def(py::init<const numpy_value_type &, const stateStr &>())
         .def(py::init<const _sepState &>())
         .def(py::init<const ttns::sepState<real_type> &>())
         .def("assign", [](_sepState &self, const _sepState &o)
              { self = o; })
         .def("assign", [](_sepState &self, const ttns::sepState<real_type> &o)
              { self = o; })
         .def("clear", &_sepState::clear, "For details see :meth:`pyttn.sepState_dtype.clear`")
         .def("resize", &_sepState::resize, "For details see :meth:`pyttn.sepState_dtype.resize`")
         .def_property("coeff", static_cast<const numpy_value_type &(_sepState::*)() const>(&_sepState::coeff), [](_sepState &o, const numpy_value_type &i)
                       { o.coeff() = value_type(i); })
         .def("size", &_sepState::size, "For details see :meth:`pyttn.sepState_dtype.size`")
         .def("nnz", &_sepState::nnz, "For details see :meth:`pyttn.sepState_dtype.nnz`")
         .def("state", [](_sepState &o)
              { return o.state(); }, "For details see :meth:`pyttn.sepState_dtype.state`")
         //.def("__str__", [](const _sepState &o){std::ostringstream oss; oss << o; return oss.str();})

         .def("__eq__", [](const _sepState &a, const _sepState &b)
              { return a == b; })
         .def("__ne__", [](const _sepState &a, const _sepState &b)
              { return !(a == b); })
         .def("__mul__", [](const _sepState &a, const real_type &b)
              { return a * b; })
         .def("__mul__", [](const _sepState &a, const numpy_value_type &b)
              { return a * value_type(b); })
         .def("__mul__", [](const _sepState &a, const numpy_complex_type &b)
              { return a * complex_type(b); })
         .def("__rmul__", [](const _sepState &a, const real_type &b)
              { return a * b; })
         .def("__rmul__", [](const _sepState &a, const numpy_value_type &b)
              { return a * value_type(b); })
         .def("__rmul__", [](const _sepState &a, const numpy_complex_type &b)
              { return a * complex_type(b); })
         .def("__imul__", [](_sepState &a, const real_type &b)
              { return a *= b; })
         .def("__imul__", [](_sepState &a, const numpy_value_type &b)
              { return a *= value_type(b); })

         .def("__truediv__", [](const _sepState &a, const real_type &b)
              { return a / b; })
         .def("__truediv__", [](const _sepState &a, const numpy_value_type &b)
              { return a / value_type(b); })
         .def("__truediv__", [](const _sepState &a, const numpy_complex_type &b)
              { return a / complex_type(b); })
         .def("__itruediv__", [](_sepState &a, const real_type &b)
              { return a /= b; })
         .def("__itruediv__", [](_sepState &a, const numpy_value_type &b)
              { return a /= value_type(b); })

         .def("__add__", [](_sepState &a, _sepState &b)
              { return a + b; })
         .def("__add__", [](_sepState &a, ttns::sepState<real_type> &b)
              { return a + b; })
         .def("__add__", [](_sepState &a, ttns::sepState<complex_type> &b)
              { return a + b; })
         .def("__add__", [](_sepState &a, stateStr &b)
              { return a + b; })
         .def("__add__", [](_sepState &a, ttns::ket<T> &b)
              { return a - b; })
         .def("__add__", [](_sepState &a, ttns::ket<real_type> &b)
              { return a - b; })
         .def("__add__", [](_sepState &a, ttns::ket<complex_type> &b)
              { return a - b; })

         .def("__sub__", [](_sepState &a, _sepState &b)
              { return a - b; })
         .def("__sub__", [](_sepState &a, ttns::sepState<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_sepState &a, ttns::sepState<complex_type> &b)
              { return b - a; })
         .def("__sub__", [](_sepState &a, stateStr &b)
              { return a - b; })
         .def("__rsub__", [](_sepState &a, stateStr &b)
              { return b - a; })
         .def("__sub__", [](_sepState &a, ttns::ket<T> &b)
              { return a - b; })
         .def("__sub__", [](_sepState &a, ttns::ket<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_sepState &a, ttns::ket<complex_type> &b)
              { return a - b; });

     using _ket = ttns::ket<T>;
     // wrapper for the state type
     py::class_<_ket>(m, ("ket_" + label).c_str())
         .def(py::init())
         .def(py::init<const _ket &>())
         .def(py::init<const ttns::ket<real_type> &>())

         .def("assign", [](_ket &self, const _ket &o)
              { self = o; })
         .def("assign", [](_ket &self, const ttns::ket<real_type> &o)
              { self = o; })
         .def("clear", &_ket::clear, "For details see :meth:`pyttn.ket_dtype.clear`")
         .def("reserve", &_ket::reserve, "For details see :meth:`pyttn.ket_dtype.reserve`")
         .def("nterms", &_ket::nterms, "For details see :meth:`pyttn.ket_dtype.nterms`")
         .def("contains", &_ket::contains, "For details see :meth:`pyttn.ket_dtype.contains`")
         .def("__setitem__", [](_ket &o, const stateStr &i, const T &v)
              { o[i] = v; })
         .def("__getitem__", [](_ket &o, const stateStr &i)
              {
                    if(!o.contains(i)){RAISE_EXCEPTION("Failed to get element.  Index out of bounds.");}
                    return o[i]; })
         .def("__str__", [](const _ket &o)
              {std::ostringstream oss; oss << o; return oss.str(); })

         .def("insert", static_cast<void (_ket::*)(const _sepState &)>(&_ket::insert))
         .def("prune_zeros", &_ket::prune_zeros, py::arg("tol") = 1e-15)

         .def("__iter__", [](_ket &s)
              { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())

         .def("__imul__", [](_ket &a, const real_type &b)
              { return a *= b; })
         .def("__imul__", [](_ket &a, const numpy_value_type &b)
              { return a *= value_type(b); })
         .def("__itruediv__", [](_ket &a, const real_type &b)
              { return a /= b; })
         .def("__itruediv__", [](_ket &a, const numpy_value_type &b)
              { return a /= value_type(b); })

         .def("__iadd__", [](_ket &a, stateStr &b)
              { return a += b; })
         .def("__iadd__", [](_ket &a, _sepState &b)
              { return a += b; })
         .def("__iadd__", [](_ket &a, _ket &b)
              { return a += b; })

         .def("__isub__", [](_ket &a, stateStr &b)
              { return a -= b; })
         .def("__isub__", [](_ket &a, _sepState &b)
              { return a -= b; })
         .def("__isub__", [](_ket &a, _ket &b)
              { return a -= b; })

         .def("__add__", [](_ket &a, _sepState &b)
              { return a + b; })
         .def("__add__", [](_ket &a, stateStr &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::sepState<real_type> &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::sepState<complex_type> &b)
              { return a + b; })
         .def("__add__", [](_ket &a, _ket &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::ket<real_type> &b)
              { return a + b; })
         .def("__add__", [](_ket &a, ttns::ket<complex_type> &b)
              { return a + b; })

         .def("__sub__", [](_ket &a, _sepState &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, stateStr &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::sepState<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::sepState<complex_type> &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, _ket &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::ket<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_ket &a, ttns::ket<complex_type> &b)
              { return a - b; });
}

template <typename T>
void initialise_state(py::module &m);

#endif
