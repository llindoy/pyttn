/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
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

#include "symbolic_transpose.hpp"
#include "../../pyttn_typedef.hpp"

namespace py = pybind11;

void initialise_symbolic_transpose(py::module &m)
{
  using namespace ttns;
  using real_type = pyttn_real_type;

  using complex_type = std::complex<real_type>;
  using _T = typename linalg::numpy_converter<complex_type>::type;

#ifdef BUILD_REAL_TTN
  using opdictr = operator_dictionary<real_type, linalg::blas_backend>;
#endif
  using opdictc = operator_dictionary<complex_type, linalg::blas_backend>;

#ifdef BUILD_REAL_TTN
  using opdictr_gpu = operator_dictionary<real_type, linalg::cuda_backend>;
#endif
  using opdictc_gpu = operator_dictionary<complex_type, linalg::cuda_backend>;

  py::class_<symbolic_transpose>(m, "symbolic_transpose")
      .def_static(
          "apply",
          static_cast<void (*)(const sOP &, const system_modes &, sNBO<complex_type> &)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sPOP &, const system_modes &, sNBO<complex_type> &)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sNBO<complex_type> &, const system_modes &, sNBO<complex_type> &)>(&symbolic_transpose::apply)
        )    
        .def_static(
          "apply",
          static_cast<void (*)(const sSOP<complex_type> &, const system_modes &, sSOP<complex_type> &)>(&symbolic_transpose::apply)
        )    
      .def_static(
          "apply",
          static_cast<void (*)(const sOP &, const opdictc&, const system_modes &, sNBO<complex_type> &, opdictc&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sPOP &, const opdictc&, const system_modes &, sNBO<complex_type> &, opdictc&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sNBO<complex_type> &, const opdictc&, const system_modes &, sNBO<complex_type> &, opdictc&)>(&symbolic_transpose::apply)
        )    
        .def_static(
          "apply",
          static_cast<void (*)(const sSOP<complex_type> &, const opdictc&, const system_modes &, sSOP<complex_type> &, opdictc&)>(&symbolic_transpose::apply)
        )    
      .def_static(
          "apply",
          static_cast<void (*)(const sOP &, const opdictc_gpu&, const system_modes &, sNBO<complex_type> &, opdictc_gpu&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sPOP &, const opdictc_gpu&, const system_modes &, sNBO<complex_type> &, opdictc_gpu&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sNBO<complex_type> &, const opdictc_gpu&, const system_modes &, sNBO<complex_type> &, opdictc_gpu&)>(&symbolic_transpose::apply)
        )    
        .def_static(
          "apply",
          static_cast<void (*)(const sSOP<complex_type> &, const opdictc_gpu&, const system_modes &, sSOP<complex_type> &, opdictc_gpu&)>(&symbolic_transpose::apply)
        )    
// Functions for handling real valued SOPs.  These should only be allowed if the user has compiled with the option BUILD_REAL_TTN
        .def_static(
          "apply",
          static_cast<void (*)(const sNBO<real_type> &, const system_modes &, sNBO<real_type> &)>(&symbolic_transpose::apply)
        )    
        .def_static(
          "apply",
          static_cast<void (*)(const sSOP<real_type> &, const system_modes &, sSOP<real_type> &)>(&symbolic_transpose::apply)
        )    
      .def_static(
          "apply",
          static_cast<void (*)(const sOP &, const opdictr&, const system_modes &, sNBO<real_type> &, opdictr&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sPOP &, const opdictr&, const system_modes &, sNBO<complex_type> &, opdictr&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sNBO<real_type> &, const opdictr&, const system_modes &, sNBO<real_type> &, opdictr&)>(&symbolic_transpose::apply)
        )    
        .def_static(
          "apply",
          static_cast<void (*)(const sSOP<real_type> &, const opdictr&, const system_modes &, sSOP<real_type> &, opdictr&)>(&symbolic_transpose::apply)
        )    
      .def_static(
          "apply",
          static_cast<void (*)(const sOP &, const opdictr_gpu&, const system_modes &, sNBO<real_type> &, opdictr_gpu&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sPOP &, const opdictr_gpu&, const system_modes &, sNBO<real_type> &, opdictr_gpu&)>(&symbolic_transpose::apply)
        )
        .def_static(
          "apply",
          static_cast<void (*)(const sNBO<real_type> &, const opdictr_gpu&, const system_modes &, sNBO<real_type> &, opdictr_gpu&)>(&symbolic_transpose::apply)
        )    
        .def_static(
          "apply",
          static_cast<void (*)(const sSOP<real_type> &, const opdictr_gpu&, const system_modes &, sSOP<real_type> &, opdictr_gpu&)>(&symbolic_transpose::apply)
        ) 
      ;
}



