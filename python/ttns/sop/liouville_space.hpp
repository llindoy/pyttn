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

#ifndef PYTHON_BINDING_LIOUVILLE_SPACE_SUPEROPERATORS_HPP
#define PYTHON_BINDING_LIOUVILLE_SPACE_SUPEROPERATORS_HPP

#include "../../utils.hpp"

#include <ttns_lib/sop/sSOP.hpp>
#include <ttns_lib/sop/SOP.hpp>
#include <ttns_lib/sop/liouville_space.hpp>
#include <ttns_lib/sop/operator_dictionaries/default_operator_dictionaries.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template <typename real_type>
void initialise_liouville_space(py::module &m)
{
  using namespace ttns;
  using complex_type = linalg::complex<real_type>;
  using _T = typename linalg::numpy_converter<complex_type>::type;

#ifdef BUILD_REAL_TTN
  using opdictr = operator_dictionary<real_type, linalg::blas_backend>;
#endif
  using opdictc = operator_dictionary<complex_type, linalg::blas_backend>;

#ifdef PYTTN_BUILD_CUDA
#ifdef BUILD_REAL_TTN
  using opdictr_gpu = operator_dictionary<real_type, linalg::cuda_backend>;
#endif
  using opdictc_gpu = operator_dictionary<complex_type, linalg::cuda_backend>;
#endif

  py::class_<liouville_space>(m, "liouville_space")
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, SOP<complex_type> &, _T)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, SOP<complex_type> &, _T)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, SOP<complex_type> &, _T)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, SOP<complex_type> &, _T)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc &, SOP<complex_type> &, opdictc &, _T)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc &, SOP<complex_type> &, opdictc &, _T)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc &, SOP<complex_type> &, opdictc &, _T)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc &, SOP<complex_type> &, opdictc &, _T)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
#ifdef PYTTN_BUILD_CUDA
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc_gpu &, SOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc_gpu &, SOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc_gpu &, SOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const SOP<complex_type> &, const system_modes &, const opdictc_gpu &, SOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
#endif
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, sSOP<complex_type> &, _T)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, sSOP<complex_type> &, _T)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, sSOP<complex_type> &, _T)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, sSOP<complex_type> &, _T)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc &, sSOP<complex_type> &, opdictc &, _T)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc &, sSOP<complex_type> &, opdictc &, _T)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc &, sSOP<complex_type> &, opdictc &, _T)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc &, sSOP<complex_type> &, opdictc &, _T)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
#ifdef PYTTN_BUILD_CUDA
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc_gpu &, sSOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc_gpu &, sSOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc_gpu &, sSOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, const opdictc_gpu &, sSOP<complex_type> &, opdictc_gpu &, _T)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = _T(1))
#endif
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, sSOP<real_type> &, real_type)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, sSOP<real_type> &, real_type)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, sSOP<real_type> &, real_type)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, sSOP<real_type> &, real_type)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))

// Functions for handling real valued SOPs.  These should only be allowed if the user has compiled with the option BUILD_REAL_TTN
#ifdef BUILD_REAL_TTN
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, SOP<real_type> &, real_type)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, SOP<real_type> &, real_type)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, SOP<real_type> &, real_type)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, SOP<real_type> &, real_type)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr &, SOP<real_type> &, opdictr &, real_type)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr &, SOP<real_type> &, opdictr &, real_type)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr &, SOP<real_type> &, opdictr &, real_type)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr &, SOP<real_type> &, opdictr &, real_type)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr &, sSOP<real_type> &, opdictr &, real_type)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr &, sSOP<real_type> &, opdictr &, real_type)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr &, sSOP<real_type> &, opdictr &, real_type)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr &, sSOP<real_type> &, opdictr &, real_type)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
#ifdef PYTTN_BUILD_CUDA
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr_gpu &, SOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr_gpu &, SOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr_gpu &, SOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const SOP<real_type> &, const system_modes &, const opdictr_gpu &, SOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "left_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr_gpu &, sSOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::left_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "right_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr_gpu &, sSOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::right_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "commutator_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr_gpu &, sSOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::commutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
      .def_static(
          "anticommutator_superoperator",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, const opdictr_gpu &, sSOP<real_type> &, opdictr_gpu &, real_type)>(&liouville_space::anticommutator_superoperator),
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff") = real_type(1))
#endif
#endif
      ;
}
#endif
