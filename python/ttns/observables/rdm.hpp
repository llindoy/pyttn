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

#ifndef PYTHON_BINDING_TTNS_RDM_HPP
#define PYTHON_BINDING_TTNS_RDM_HPP

#include <ttns_lib/observables/rdm.hpp>
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
void init_rdm(py::module &m, const std::string &label)
{
  using namespace ttns;

  using matrix_type = linalg::matrix<T, backend>;
  using _rdm = rdm<T, backend>;
  using _ttn = ttn<T, backend>;

  py::class_<_rdm>(m, (std::string("rdm_") + label).c_str())
      .def(py::init())
      .def(py::init<const _ttn &, bool>(), py::arg(), py::arg("two_body") = false)

      .def("assign", [](_rdm &op, const _rdm &o)
           { return op = o; })
      .def("__copy__", [](const _rdm &o)
           { return _rdm(o); })
      .def("__deepcopy__", [](const _rdm &o, py::dict)
           { return _rdm(o); }, py::arg("memo"))
      .def("clear", &_rdm::clear)

      /*
       *  Functions for handling ttns
       */
      .def("resize", &_rdm::template resize<_ttn>, py::arg(), py::arg("two_body") = false)
      .def("__call__", [](_rdm &o, _ttn& A, size_t m1, matrix_type& mat){o(A, m1, mat);})
      .def("__call__", [](_rdm &o, _ttn& A, size_t m1){matrix_type mat; o(A, m1, mat); return mat;})
      .def("__call__", [](_rdm &o, _ttn& A, size_t m1, size_t m2, matrix_type& mat){o(A, m1, m2, mat);})
      .def("__call__", [](_rdm &o, _ttn& A, size_t m1, size_t m2){matrix_type mat; o(A, m1, m2, mat); return mat;})

      .def("backend", [](const _rdm &)
           { return linalg::traits<backend>::label(); });
}

template <typename real_type, typename backend>
void initialise_rdm(py::module &m)
{
  using complex_type = std::complex<real_type>;

#ifdef BUILD_REAL_TTN
  init_rdm<real_type, backend>(m, "real");
#endif
  init_rdm<complex_type, backend>(m, "complex");
}
#endif // PYTHON_BINDING_TTNS_RDM_HPP
