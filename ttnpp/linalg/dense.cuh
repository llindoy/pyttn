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

#ifndef PYTTN_LINALG_DENSE_CUH_
#define PYTTN_LINALG_DENSE_CUH_

/**
 *  Class includes all of the required headers to use the dense linear algebra component of the linalg library
 */
#include "backends/cuda/cuda_backend.hpp"
#include "tensor/dense/cuda/tensor.cuh"
#include "tensor/dense/cuda/tensor_view.cuh"
#include "tensor/dense/cuda/tensor_slice.cuh"
#include "tensor/dense/dense_typedefs.hpp"
#include "algebra/algebra.cuh"

namespace linalg
{
    template <typename T, size_t D>
    using device_tensor = tensor<T, D, cuda_backend>;
    template <typename T>
    using device_matrix = tensor<T, 2, cuda_backend>;
    template <typename T>
    using device_vector = tensor<T, 1, cuda_backend>;
}

#endif // PYTTN_LINALG_DENSE_CUH_
