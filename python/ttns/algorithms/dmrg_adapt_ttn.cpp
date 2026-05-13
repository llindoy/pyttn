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

#include "dmrg.tpp"
#include "../../pyttn_typedef.hpp"

void initialise_dmrg_adaptive_ttn(py::module &m)
{
    using complex_type = std::complex<pyttn_real_type>;
    init_dmrg_adaptive<complex_type, ttns::ttn, linalg::blas_backend>(m, std::string("adaptive_one_site_dmrg_complex"));
#ifdef BUILD_REAL_TTN
    init_dmrg_adaptive<pyttn_real_type, ttns::ttn, linalg::blas_backend>(m, std::string("adaptive_one_site_dmrg_real"));
#endif
}