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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_CSR_MATRIX_CUH_
#define PYTTN_LINALG_TENSOR_SPARSE_CSR_MATRIX_CUH_

#include <vector>
#include <tuple>
#include <algorithm>

#include "../../../backends/cuda/cuda_backend.hpp"
#include "../../../utils/memory_helper.cuh"
#include "../csr_matrix.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/array.hpp>
#endif

namespace linalg
{
    template <typename T>
    class csr_matrix<T, cuda_backend> : public csr_matrix_base<csr_matrix<T, cuda_backend>>
    {
    public:
        using self_type = csr_matrix<T, cuda_backend>;
        using base_type = csr_matrix_base<self_type>;
        using real_type = typename base_type::real_type;

        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using index_pointer = typename base_type::index_pointer;
        using const_index_pointer = typename base_type::const_index_pointer;
        using coo_type = typename base_type::coo_type;

    public:
        template <typename... Args>
        csr_matrix(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct csr matrix object.");
        }
        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

        inline void transpose(csr_matrix<T, cuda_backend> &o, T alpha = T(1)) const
        {
            RAISE_EXCEPTION("CUDA CSR MATRIX TRANSPOSE NOT IMPLEMENTED.");
        }

        inline tensor<T, 2, blas_backend> todense() const
        {
            csr_matrix<T, blas_backend> ret(*this);
            return ret.todense();
        }
    }; // csr_matrix<T, blas_backend>

    template <typename T>
    std::ostream &operator<<(std::ostream &out, const csr_matrix<T, cuda_backend> &_mat)
    {
        csr_matrix<T, blas_backend> mat(_mat);
        out << mat;
        return out;
    }

} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_CSR_MATRIX_CUH_//
