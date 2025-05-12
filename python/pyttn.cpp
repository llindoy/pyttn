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

#define TTNS_REGISTER_COMPLEX_DOUBLE_OPERATOR

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pyttn_typedef.hpp"

#include "linalg/backend.hpp"
#include "linalg/orthogonal_vector_gen.hpp"
#include "linalg/sparseMatrix.hpp"
#include "linalg/tensor.hpp"

#include "utils/discretisation.hpp"
#include "utils/orthopol.hpp"

#include "ttns/sop/state.hpp"
#include "ttns/sop/SOP.hpp"
#include "ttns/sop/liouville_space.hpp"
#include "ttns/sop/operator_dictionary.hpp"
#include "ttns/sop/sSOP.hpp"
#include "ttns/sop/system_information.hpp"
#include "ttns/sop/toDense.hpp"

#include "ttns/sop/models/models.hpp"

#include "ttns/ttn/ntree.hpp"
#include "ttns/ttn/ttn.hpp"

#include "ttns/ttn/ms_ttn.hpp"

#include "ttns/observables/matrix_element.hpp"

#include "ttns/operators/product_operator.hpp"
#include "ttns/operators/siteOperators.hpp"
#include "ttns/operators/sop_operator.hpp"

#include "ttns/algorithms/dmrg.hpp"
#include "ttns/algorithms/tdvp.hpp"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(ttnpp, m)
{
    m.doc() = R"mydelimiter(
      Python wrapping of the TTNS_LIB library for performing calculations on tree tensor network states
      )mydelimiter";
    auto m_linalg = m.def_submodule("linalg", R"mydelimiter(
        Linear algebra submodule for TTNS library."
        )mydelimiter");
    auto m_orthopol = m.def_submodule("utils", R"mydelimiter(
        Orthogonal polynomials submodule for TTNS library.
        )mydelimiter");

    auto m_models = m.def_submodule("models", R"mydelimiter(
        Submodule of the TTNS Library containing pre-defined models specifying system and Hamiltonian information.
        )mydelimiter");

    auto m_ops = m.def_submodule("ops", R"mydelimiter(
        Operator submodule for TTNS library.
        )mydelimiter");

#ifdef PYTTN_BUILD_CUDA
    auto m_cuda = m.def_submodule("cuda", R"mydelimiter(
        Submodule containing all cuda accelerated pyTTN classes."
        )mydelimiter");

    auto m_linalg_gpu = m_cuda.def_submodule("linalg", R"mydelimiter(
        Linear algebra submodule for TTNS library."
        )mydelimiter");

    auto m_models_gpu = m_cuda.def_submodule("models", R"mydelimiter(
        Pre-defined models specifying system and Hamiltonian information.
        )mydelimiter");
    auto m_ops_gpu = m_cuda.def_submodule("ops", R"mydelimiter(
        Operator submodule for TTNS library.
        )mydelimiter");

#endif

    //
    // Wrap the required linear algebra types to enable python based instantiation
    // of operators.
    //
    // The CPU implementations
    initialise_tensors<pyttn_real_type>(m_linalg);
    initialise_sparse_matrices<pyttn_real_type>(m_linalg);
    initialise_orthogonal_vector<pyttn_real_type, linalg::blas_backend>(m_linalg);

    // initialise the blas backend objects
    initialise_blas_backend(m_linalg);

#ifdef PYTTN_BUILD_CUDA
    // the GPU implementations
    initialise_tensors_cuda<pyttn_real_type>(m_linalg_gpu);
    initialise_sparse_matrices_cuda<pyttn_real_type>(m_linalg_gpu);
    initialise_orthogonal_vector<pyttn_real_type, linalg::cuda_backend>(m_linalg_gpu);

    // initialise the cuda environment objects
    initialise_cuda_backend(m_linalg_gpu);
#endif

    //
    // Wrap the required utils functions
    //
    initialise_orthopol<pyttn_real_type>(m_orthopol);
    initialise_discretisation<pyttn_real_type>(m_orthopol);

    //
    // Wrap the sOP functionality
    //
    initialise_sSOP<pyttn_real_type>(m);
    initialise_system_info(m);
    initialise_state<pyttn_real_type>(m);
    initialise_SOP<pyttn_real_type>(m);
    initialise_operator_dictionary<pyttn_real_type, linalg::blas_backend>(m);
    initialise_liouville_space<pyttn_real_type>(m);
    initialise_convert_to_dense<pyttn_real_type>(m);

    //
    // Wrap the models functionality included in SOP
    //
    initialise_models<pyttn_real_type>(m_models);

#ifdef PYTTN_BUILD_CUDA
    // the GPU implementations
    initialise_operator_dictionary<pyttn_real_type, linalg::cuda_backend>(m_cuda);
#endif

    //
    // Wrap core ttns functionality
    //
    initialise_ntree(m);
    initialise_ttn<pyttn_real_type, linalg::blas_backend>(m);
    initialise_msttn<pyttn_real_type, linalg::blas_backend>(m);

#ifdef PYTTN_BUILD_CUDA
    initialise_ttn<pyttn_real_type, linalg::cuda_backend>(m_cuda);
    initialise_msttn<pyttn_real_type, linalg::cuda_backend>(m_cuda);
#endif

    initialise_matrix_element<pyttn_real_type, linalg::blas_backend>(m);

    initialise_site_operators<pyttn_real_type, linalg::blas_backend>(m_ops);
    initialise_product_operator<pyttn_real_type, linalg::blas_backend>(m);
    initialise_sop_operator<pyttn_real_type, linalg::blas_backend>(m);

#ifdef PYTTN_BUILD_CUDA
    initialise_matrix_element<pyttn_real_type, linalg::cuda_backend>(m_cuda);

    initialise_site_operators<pyttn_real_type, linalg::cuda_backend>(m_ops_gpu);
    initialise_product_operator<pyttn_real_type, linalg::cuda_backend>(m_cuda);
    initialise_sop_operator<pyttn_real_type, linalg::cuda_backend>(m_cuda);
#endif

    //
    // Wrap the core algorithms for operating on ttns
    //
    initialise_dmrg<pyttn_real_type, linalg::blas_backend>(m);
    initialise_dmrg_adaptive<pyttn_real_type, linalg::blas_backend>(m);

    initialise_tdvp<pyttn_real_type, linalg::blas_backend>(m);
    initialise_tdvp_adaptive<pyttn_real_type, linalg::blas_backend>(m);

#ifdef PYTTN_BUILD_CUDA
    initialise_dmrg<pyttn_real_type, linalg::cuda_backend>(m);
    // initialise_dmrg_adaptive<pyttn_real_type, linalg::cuda_backend>(m);

    initialise_tdvp<pyttn_real_type, linalg::cuda_backend>(m);
    // initialise_tdvp_adaptive<pyttn_real_type, linalg::cuda_backend>(m);
#endif
}
