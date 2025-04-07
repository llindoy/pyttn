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

#ifndef PYTHON_BINDING_MAPPED_SOP_HPP
#define PYTHON_BINDING_MAPPED_SOP_HPP

#include <ttns_lib/sop/sSOP.hpp>
#include <ttns_lib/sop/SOP.hpp>
#include <ttns_lib/sop/multiset_SOP.hpp>

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
void init_SOP(py::module &m, const std::string &label)
{
     using namespace ttns;

     using _T = typename linalg::numpy_converter<T>::type;

     using real_type = typename linalg::get_real_type<T>::type;
     using _SOP = SOP<T>;
     using _msSOP = multiset_SOP<T>;
     // wrapper for the sPOP type
     py::class_<_SOP>(m, label.c_str())
         .def(py::init<size_t>())
         .def(py::init<size_t, const std::string &>())
         .def(py::init<const _SOP &>())
         .def("assign", [](_SOP &self, const _SOP &o)
              { self = o; })
         .def("__copy__", [](const _SOP &o)
              { return _SOP(o); })
         .def("__deepcopy__", [](const _SOP &o, py::dict)
              { return _SOP(o); }, py::arg("memo"))

         .def("__iter__", [](_SOP &s)
              { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())
         .def("clear", &_SOP::clear)
         .def("resize", &_SOP::resize)
         .def("reserve", &_SOP::reserve)
         .def("nmodes", &_SOP::nmodes)
         .def("nterms", &_SOP::nterms)
         .def_property("operator_dictionary", &_SOP::operator_dictionary, &_SOP::set_operator_dictionary)
         .def("set_operator_dictionary", &_SOP::set_operator_dictionary)
         .def("get_operator_dictionary", &_SOP::operator_dictionary)

         .def("insert", static_cast<void (_SOP::*)(const T &, const sPOP &)>(&_SOP::insert))
         .def("insert", static_cast<void (_SOP::*)(const sNBO<T> &)>(&_SOP::insert))

         .def("set_is_fermion_mode", &_SOP::set_is_fermionic_mode)
         .def("prune_zeros", &_SOP::prune_zeros, py::arg("tol") = 1e-15)
         .def("jordan_wigner", static_cast<_SOP &(_SOP::*)(const system_modes &, double)>(&_SOP::jordan_wigner), py::arg(), py::arg("tol") = 1e-15)

         .def("expand", &_SOP::expand)

         .def_property("label", static_cast<const std::string &(_SOP::*)() const>(&_SOP::label), [](_SOP &o, const std::string &i)
                       { o.label() = i; })
         .def("__str__", [](const _SOP &o)
              {std::ostringstream oss; oss << o; return oss.str(); })

         .def("__imul__", [](_SOP &a, const real_type &b)
              { return a *= b; })
         .def("__imul__", [](_SOP &a, const _T &b)
              { return a *= T(b); })
         .def("__idiv__", [](_SOP &a, const real_type &b)
              { return a /= b; })
         .def("__idiv__", [](_SOP &a, const _T &b)
              { return a /= T(b); })

         .def("__iadd__", [](_SOP &a, const real_type &b)
              { return a += b; })
         .def("__iadd__", [](_SOP &a, const _T &b)
              { return a += T(b); })
         .def("__iadd__", [](_SOP &a, const sOP &b)
              { return a += b; })
         .def("__iadd__", [](_SOP &a, const sPOP &b)
              { return a += b; })
         .def("__iadd__", [](_SOP &a, const sNBO<real_type> &b)
              { return a += b; })
         .def("__iadd__", [](_SOP &a, const sNBO<T> &b)
              { return a += b; })
         .def("__iadd__", [](_SOP &a, const sSOP<real_type> &b)
              { return a += b; })
         .def("__iadd__", [](_SOP &a, const sSOP<T> &b)
              { return a += b; })

         .def("__isub__", [](_SOP &a, const real_type &b)
              { return a -= b; })
         .def("__isub__", [](_SOP &a, const _T &b)
              { return a -= T(b); })
         .def("__isub__", [](_SOP &a, const sOP &b)
              { return a -= b; })
         .def("__isub__", [](_SOP &a, const sPOP &b)
              { return a -= b; })
         .def("__isub__", [](_SOP &a, const sNBO<real_type> &b)
              { return a -= b; })
         .def("__isub__", [](_SOP &a, const sNBO<T> &b)
              { return a -= b; })
         .def("__isub__", [](_SOP &a, const sSOP<real_type> &b)
              { return a -= b; })
         .def("__isub__", [](_SOP &a, const sSOP<T> &b)
              { return a -= b; })

         .def("__add__", [](_SOP &a, const _T &b)
              { return a + T(b); })
         .def("__add__", [](_SOP &a, const real_type &b)
              { return a + b; })
         .def("__add__", [](_SOP &a, const sOP &b)
              { return a + b; })
         .def("__add__", [](_SOP &a, const sPOP &b)
              { return a + b; })
         .def("__add__", [](_SOP &a, const sNBO<real_type> &b)
              { return a + b; })
         .def("__add__", [](_SOP &a, const sNBO<T> &b)
              { return a + b; })
         .def("__add__", [](_SOP &a, const sSOP<real_type> &b)
              { return a + b; })
         .def("__add__", [](_SOP &a, const sSOP<T> &b)
              { return a + b; })

         .def("__radd__", [](_SOP &b, const _T &a)
              { return T(a) + b; })
         .def("__radd__", [](_SOP &b, const real_type &a)
              { return a + b; })
         .def("__radd__", [](_SOP &b, const sOP &a)
              { return a + b; })
         .def("__radd__", [](_SOP &b, const sPOP &a)
              { return a + b; })
         .def("__radd__", [](_SOP &b, const sNBO<real_type> &a)
              { return a + b; })
         .def("__radd__", [](_SOP &b, const sNBO<T> &a)
              { return a + b; })
         .def("__radd__", [](_SOP &b, const sSOP<real_type> &a)
              { return a + b; })
         .def("__radd__", [](_SOP &b, const sSOP<T> &a)
              { return a + b; })

         .def("__sub__", [](_SOP &a, const _T &b)
              { return a - T(b); })
         .def("__sub__", [](_SOP &a, const real_type &b)
              { return a - b; })
         .def("__sub__", [](_SOP &a, const sOP &b)
              { return a - b; })
         .def("__sub__", [](_SOP &a, const sPOP &b)
              { return a - b; })
         .def("__sub__", [](_SOP &a, const sNBO<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_SOP &a, const sNBO<T> &b)
              { return a - b; })
         .def("__sub__", [](_SOP &a, const sSOP<real_type> &b)
              { return a - b; })
         .def("__sub__", [](_SOP &a, const sSOP<T> &b)
              { return a - b; })
         .def("__rsub__", [](_SOP &b, const _T &a)
              { return T(a) - b; })
         .def("__rsub__", [](_SOP &b, const real_type &a)
              { return a - b; })
         .def("__rsub__", [](_SOP &b, const sOP &a)
              { return a - b; })
         .def("__rsub__", [](_SOP &b, const sPOP &a)
              { return a - b; })
         .def("__rsub__", [](_SOP &b, const sNBO<real_type> &a)
              { return a - b; })
         .def("__rsub__", [](_SOP &b, const sNBO<T> &a)
              { return a - b; })
         .def("__rsub__", [](_SOP &b, const sSOP<real_type> &a)
              { return a - b; })
         .def("__rsub__", [](_SOP &b, const sSOP<T> &a)
              { return a - b; })
         .doc() = R"mydelim(
            A class for storing a compact representation of a sum-of-product string operators.  This class requires
            knowledge of the total number of degrees of freedom.

            Construct arguments

            :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
            :type A: ttn_complex
            :param H: The Hamiltonian sop operator object
            :type H: sop_operator_complex
            :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
            :type krylov_dim: int, optional
            :param numthreads: The number of openmp threads to be used by the solver. (Default: 1)
            :type numthreads: int, optional

            Callable arguments

            :param A: Tree Tensor Network that the DMRG algorithm will act on
            :type A: ttn_complex
            :param h: The Hamiltonian sop operator object
            :type h: sop_operator_complex
            :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
            :type update_env: bool, optional
          )mydelim";

     // wrapper for the sPOP type
     py::class_<_msSOP>(m, (std::string("multiset_") + label).c_str())
         .def(py::init())
         .def(py::init<size_t, size_t>())
         .def(py::init<size_t, size_t, const std::string &>())
         .def(py::init<const _msSOP &>())
         .def("assign", [](_msSOP &self, const _msSOP &o)
              { self = o; })
         .def("__copy__", [](const _msSOP &o)
              { return _msSOP(o); })
         .def("__deepcopy__", [](const _msSOP &o, py::dict)
              { return _msSOP(o); }, py::arg("memo"))
         .def("clear", &_msSOP::clear)
         .def("resize", &_msSOP::resize)
         .def("nmodes", &_msSOP::nmodes)
         .def("nset", &_msSOP::nset)
         .def("nterms", &_msSOP::nterms)

         .def("set", static_cast<void (_msSOP::*)(size_t, size_t, const SOP<T> &)>(&_msSOP::set))

         .def("set_is_fermion_mode", &_msSOP::set_is_fermionic_mode)
         .def("jordan_wigner", static_cast<_msSOP &(_msSOP::*)(const system_modes &, double)>(&_msSOP::jordan_wigner), py::arg(), py::arg("tol") = 1e-15)

         .def("__getitem__", [](_msSOP &i, std::pair<size_t, size_t> ind) -> _SOP &
              { return i(std::get<0>(ind), std::get<1>(ind)); }, py::return_value_policy::reference)
         .def("__setitem__", [](_msSOP &i, std::pair<size_t, size_t> ind, const _SOP &o)
              { i(std::get<0>(ind), std::get<1>(ind)) = o; })
         .def_property("label", static_cast<const std::string &(_msSOP::*)() const>(&_msSOP::label), [](_msSOP &o, const std::string &i)
                       { o.label() = i; })
         .def("__str__", [](const _msSOP &o)
              {std::ostringstream oss; oss << o; return oss.str(); });

     // SOP<T>& operator()(size_t i, size_t j)
     // const SOP<T>& operator()(size_t i, size_t j) const
}

template <typename T>
void initialise_SOP(py::module &m);

#endif
