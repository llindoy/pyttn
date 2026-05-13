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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_CUH_
#define PYTTN_LINALG_DECOMPOSITIONS_CUH_

/**
 *  Class includes all of the required headers to use the dense linear algebra component of the linalg library
 */
#include "backends/cuda/cuda_backend.hpp"
#include "backends/cuda/cuda_backend.cuh"
#include "backends/cuda/cuda_algebra.cuh"

#include "dense.hpp"
#include "dense.cuh"
#include "sparse.hpp"
#include "sparse.cuh"

#include "decompositions/eigensolvers/eigensolver.hpp"
#include "decompositions/eigensolvers/eigensolver.cuh"
#include "decompositions/generalised_eigensolvers/generalised_eigensolver.hpp"
#include "decompositions/lu_decomposition/lu_decomposition.hpp"
#include "decompositions/lu_decomposition/lu_decomposition.cuh"
//#include "decompositions/qr/qr.hpp"
#include "decompositions/singular_value_decomposition/singular_value_decomposition.hpp"
#include "decompositions/singular_value_decomposition/singular_value_decomposition_cuda.cuh"
#include "decompositions/sparse/arnoldi_iteration.hpp"
//#include "decompositions/tridiagonalisation/tridiagonalisation.hpp"

#endif // PYTTN_LINALG_DECOMPOSITIONS_HPP_
