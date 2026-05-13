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

#include "liouville_space.hpp"
#include "../../pyttn_typedef.hpp"

namespace py = pybind11;

void initialise_liouville_space(py::module &m)
{
  using namespace ttns;
  using real_type = pyttn_real_type;
  using complex_type = std::complex<real_type>;
  using _T = typename linalg::numpy_converter<complex_type>::type;

#ifdef BUILD_REAL_TTN
  using opdictr = operator_dictionary<real_type, linalg::blas_backend>;
#endif
  using opdictc = operator_dictionary<complex_type, linalg::blas_backend>;


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
#endif
      ;
}

