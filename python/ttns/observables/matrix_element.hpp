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

#ifndef PYTHON_BINDING_TTNS_MATRIX_ELEMENTS_HPP
#define PYTHON_BINDING_TTNS_MATRIX_ELEMENTS_HPP

#include <ttns_lib/observables/matrix_element.hpp>
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
void init_matrix_element(py::module &m, const std::string &label)
{
  using namespace ttns;

  using _T = typename linalg::numpy_converter<T>::type;

  using siteop = site_operator<T, backend>;
  using prodop = product_operator<T, backend>;
  using matel = matrix_element<T, backend>;
  using _ttn = ttn<T, backend>;
  using _msttn = ms_ttn<T, backend>;
  using sop_op = sop_operator<T, backend>;
  using ms_sop_op = multiset_sop_operator<T, backend>;
  using _msttn_slice = multiset_ttn_slice<T, backend, false>;

  py::class_<matel>(m, (std::string("matrix_element_") + label).c_str())
      .def(py::init())
      .def(py::init<const _ttn &, size_t, bool>(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _ttn &, const _ttn &, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _ttn &, const sop_op &, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _ttn &, const _ttn &, const sop_op &, size_t, bool>(), py::arg(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)

      .def(py::init<const _msttn_slice &, size_t, bool>(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _msttn_slice &, const _msttn_slice &, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _msttn_slice &, const sop_op &, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _msttn_slice &, const _msttn_slice &, const sop_op &, size_t, bool>(), py::arg(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)

      .def(py::init<const _msttn &, size_t, bool>(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _msttn &, const _msttn &, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _msttn &, const ms_sop_op &, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def(py::init<const _msttn &, const _msttn &, const ms_sop_op &, size_t, bool>(), py::arg(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)

      .def("assign", [](matel &op, const matel &o)
           { return op = o; })
      .def("__copy__", [](const matel &o)
           { return matel(o); })
      .def("__deepcopy__", [](const matel &o, py::dict)
           { return matel(o); }, py::arg("memo"))
      .def("clear", &matel::clear)

      /*
       *  Functions for handling ttns
       */
      .def("resize", [](matel &o, const _ttn &A, size_t nbuffers, bool use_capacity)
           { o.resize(A, nbuffers, use_capacity); }, py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def("resize", [](matel &o, const _ttn &A, const _ttn &B, size_t nbuffers, bool use_capacity)
           { o.resize(A, B, nbuffers, use_capacity); }, py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def("__call__", [](matel &o, const _ttn &A, bool use_sparsity)
           { return o(A, use_sparsity); }, py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, siteop &op, size_t mode, const _ttn &A, bool use_sparsity)
           { return _T(o(op, mode, A, use_sparsity)); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const _ttn &A, bool use_sparsity)
           { return _T(o(ops, A, use_sparsity)); }, py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const std::vector<size_t> &modes, const _ttn &A, bool use_sparsity)
           { return _T(o(ops, modes, A, use_sparsity)); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, prodop &ops, const _ttn &A, bool use_sparsity)
           { return _T(o(ops, A, use_sparsity)); }, py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, sop_op &sop, const _ttn &A)
           { return _T(o(sop, A)); })
      .def("__call__", [](matel &o, const _ttn &A, const _ttn &B)
           { return _T(o(A, B)); })
      .def("__call__", [](matel &o, siteop &op, const _ttn &A)
           { return _T(o(op, A)); })
      .def("__call__", [](matel &o, siteop &op, const _ttn &A, const _ttn &B)
           { return _T(o(op, A, B)); })
      .def("__call__", [](matel &o, siteop &op, size_t mode, const _ttn &A, const _ttn &B)
           { return _T(o(op, mode, A, B)); })
      // inline T operator()(one_body_operator<T, backend>& op, const state_type& bra, const state_type& ket)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const _ttn &A, const _ttn &B)
           { return _T(o(ops, A, B)); })
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const std::vector<size_t> &modes, const _ttn &A, const _ttn &B)
           { return _T(o(ops, modes, A, B)); })
      .def("__call__", [](matel &o, prodop &ops, const _ttn &A, const _ttn &B)
           { return _T(o(ops, A, B)); })
      .def("__call__", [](matel &o, sop_op &sop, const _ttn &A, const _ttn &B)
           { return _T(o(sop, A, B)); })

      /*
       * Functions for working with multiset ttn slices
       */
      .def("resize", [](matel &o, const _msttn_slice &A, size_t nbuffers, bool use_capacity)
           { o.resize(A, nbuffers, use_capacity); }, py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def("resize", [](matel &o, const _msttn_slice &A, const _msttn_slice &B, size_t nbuffers, bool use_capacity)
           { o.resize(A, B, nbuffers, use_capacity); }, py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def("__call__", [](matel &o, const _msttn_slice &A, bool use_sparsity)
           { return o(A, use_sparsity); }, py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, siteop &op, size_t mode, const _msttn_slice &A, bool use_sparsity)
           { return _T(o(op, mode, A, use_sparsity)); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const _msttn_slice &A, bool use_sparsity)
           { return _T(o(ops, A, use_sparsity)); }, py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const std::vector<size_t> &modes, const _msttn_slice &A, bool use_sparsity)
           { return _T(o(ops, modes, A, use_sparsity)); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, prodop &ops, const _msttn_slice &A, bool use_sparsity)
           { return _T(o(ops, A, use_sparsity)); }, py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, sop_op &sop, const _msttn_slice &A)
           { return _T(o(sop, A)); })
      .def("__call__", [](matel &o, const _msttn_slice &A, const _msttn_slice &B)
           { return _T(o(A, B)); })
      .def("__call__", [](matel &o, siteop &op, const _msttn_slice &A)
           { return _T(o(op, A)); })
      .def("__call__", [](matel &o, siteop &op, const _msttn_slice &A, const _msttn_slice &B)
           { return _T(o(op, A, B)); })
      .def("__call__", [](matel &o, siteop &op, size_t mode, const _msttn_slice &A, const _msttn_slice &B)
           { return _T(o(op, mode, A, B)); })
      // inline T operator()(one_body_operator<T, backend>& op, const state_type& bra, const state_type& ket)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const _msttn_slice &A, const _msttn_slice &B)
           { return _T(o(ops, A, B)); })
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const std::vector<size_t> &modes, const _msttn_slice &A, const _msttn_slice &B)
           { return _T(o(ops, modes, A, B)); })
      .def("__call__", [](matel &o, prodop &ops, const _msttn_slice &A, const _msttn_slice &B)
           { return _T(o(ops, A, B)); })
      .def("__call__", [](matel &o, sop_op &sop, const _msttn_slice &A, const _msttn_slice &B)
           { return _T(o(sop, A, B)); })

      /*
       *  Functions for handling multiset ttns
       */
      .def("resize", [](matel &o, const _msttn &A, size_t nbuffers, bool use_capacity)
           { o.resize(A, nbuffers, use_capacity); }, py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def("resize", [](matel &o, const _msttn &A, const _msttn &B, size_t nbuffers, bool use_capacity)
           { o.resize(A, B, nbuffers, use_capacity); }, py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
      .def("__call__", [](matel &o, const _msttn &A, bool use_sparsity)
           { return _T(o(A, use_sparsity)); }, py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, siteop &op, size_t mode, const _msttn &A, bool use_sparsity)
           { return _T(o(op, mode, A, use_sparsity)); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const std::vector<size_t> &modes, const _msttn &A, bool use_sparsity)
           { return _T(o(ops, modes, A, use_sparsity)); }, py::arg(), py::arg(), py::arg(), py::arg("use_sparsity") = true)
      .def("__call__", [](matel &o, ms_sop_op &sop, const _msttn &A)
           { return _T(o(sop, A)); })
      .def("__call__", [](matel &o, const _msttn &A, const _msttn &B)
           { return _T(o(A, B)); })
      .def("__call__", [](matel &o, siteop &op, const _msttn &A)
           { return _T(o(op, A)); })
      .def("__call__", [](matel &o, siteop &op, const _msttn &A, const _msttn &B)
           { return _T(o(op, A, B)); })
      .def("__call__", [](matel &o, siteop &op, size_t mode, const _msttn &A, const _msttn &B)
           { return _T(o(op, mode, A, B)); })
      // inline T operator()(one_body_operator<T, backend>& op, const state_type& bra, const state_type& ket)
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const _msttn &A, const _msttn &B)
           { return _T(o(ops, A, B)); })
      .def("__call__", [](matel &o, std::vector<siteop> &ops, const std::vector<size_t> &modes, const _msttn &A, const _msttn &B)
           { return _T(o(ops, modes, A, B)); })
      .def("__call__", [](matel &o, ms_sop_op &sop, const _msttn &A, const _msttn &B)
           { return _T(o(sop, A, B)); })
      .def("backend", [](const matel &)
           { return backend::label(); });

  // inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, std::vector<T>&>::type operator()(std::vector<site_operator<T, backend>>& op, size_type mode, const state_type& bra, const state_type& ket, std::vector<T>& res)
  // inline std::vector<T>& operator()(std::vector<std::vector<site_operator<T, backend>>& ops, const std::vector<size_type>& modes, const state_type& bra, const state_type& ket, std::vector<T>& res)
}

template <typename real_type, typename backend>
void initialise_matrix_element(py::module &m)
{
  using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
  init_matrix_element<real_type, backend>(m, "real");
#endif
  init_matrix_element<complex_type, backend>(m, "complex");
}
#endif // PYTHON_BINDING_TTNS_MATRIX_ELEMENTS_HPP
