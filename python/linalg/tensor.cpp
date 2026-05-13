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

#include "tensor.tpp"
#include <linalg/linalg.hpp>
#include "../pyttn_typedef.hpp"

void initialise_tensors(py::module &m)
{
    using complex_type = std::complex<pyttn_real_type>;
    init_tensor_cpu<pyttn_real_type, 1>(m, "vector_real");
    init_matrix_cpu<pyttn_real_type>(m, "matrix_real");
    init_tensor_cpu<pyttn_real_type, 3>(m, "tensor_3_real");
    init_tensor_cpu<pyttn_real_type, 4>(m, "tensor_4_real");

    init_tensor_cpu<complex_type, 1>(m, "vector_complex");
    init_matrix_cpu<complex_type>(m, "matrix_complex");
    init_tensor_cpu<complex_type, 3>(m, "tensor_3_complex");
    init_tensor_cpu<complex_type, 4>(m, "tensor_4_complex");
}


