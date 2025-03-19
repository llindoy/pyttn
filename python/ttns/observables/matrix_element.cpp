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

#include "matrix_element.hpp"
#include "../../pyttn_typedef.hpp"

template <>
void initialise_matrix_element<pyttn_real_type, linalg::blas_backend>(py::module &m);

#ifdef PYTTN_BUILD_CUDA
template <>
void initialise_matrix_element<pyttn_real_type, linalg::cuda_backend>(py::module &m);
#endif
