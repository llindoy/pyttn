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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_LU_DECOMPOSITION_CUH_
#define PYTTN_LINALG_DECOMPOSITIONS_LU_DECOMPOSITION_CUH_

#include "../decompositions_common.hpp"
#include "lu_decomposition.hpp"
#include "../../backends/cuda/cusolver_wrapper.cuh"

namespace linalg
{
    template <typename matrix_type>
    class lu_decomposition<matrix_type, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, cuda_backend>::value, void>::type>
    {
    public:
        using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
        using backend_type = typename traits<matrix_type>::backend_type;
        using size_type = typename traits<backend_type>::size_type;
        using mem_trans = memory::transfer<backend_type, backend_type>;
        using int_type = typename traits<backend_type>::int_type;

    protected:
        vector<value_type, backend_type> d_work;
        tensor<int, 1, cuda_backend> m_gpu_info;
        tensor<int, 1> m_cpu_info;

    protected:
        template <typename mat_type>
        typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, int>::type query_worksize(mat_type &A)
        {
            int_type lwork;
            CALL_AND_HANDLE(cuda_backend::getrf_buffersize(A.size(1), A.size(0), A.buffer(), A.size(1), &lwork), "Failed to query worksize for LU decomposition.");
            return lwork;
        }

    public:
        lu_decomposition() : m_gpu_info(1), m_cpu_info(1) {}

        template <typename mat_type, typename mat_typeb>
        internal::valid_decomp_matrix_type_2<matrix_type, mat_type, mat_typeb, void>
        operator()(const mat_type &A, mat_typeb &LU, vector<typename traits<backend_type>::int_type, backend_type> &ipiv)
        {
            try
            {
                m_gpu_info.resize(1);
                CALL_AND_HANDLE(internal::lu_result_validation::validate_ipiv(A, ipiv), "Failed to validate pivot array.");
                CALL_AND_HANDLE(LU = A, "Failed to copy matrix.");
                size_type lwork;
                CALL_AND_HANDLE(lwork = query_worksize(LU), "Failed to query worksize");
                CALL_AND_HANDLE(d_work.resize(lwork), "Failed to resize workspace array.");
                CALL_AND_HANDLE(cuda_backend::getrf(LU.size(1), LU.size(0), LU.buffer(), LU.size(1), d_work.buffer(), ipiv.buffer(), m_gpu_info.buffer()), "Lapack call failed.");
                m_cpu_info = m_gpu_info;
                CALL_AND_RETHROW(cusolver::getrf_error_handling(m_cpu_info(0), 'a'));
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating LU decomposition.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate LU decomposition.");
            }
        }

        template <typename mat_type>
        typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type
        operator()(mat_type &A, vector<typename traits<backend_type>::int_type, backend_type> &ipiv)
        {
            try
            {
                m_gpu_info.resize(1);
                CALL_AND_HANDLE(internal::lu_result_validation::validate_ipiv(A, ipiv), "Failed to validate pivot array.");
                size_type lwork;
                CALL_AND_HANDLE(lwork = query_worksize(A), "Failed to query worksize");
                CALL_AND_HANDLE(d_work.resize(lwork), "Failed to resize workspace array.");
                CALL_AND_HANDLE(cuda_backend::getrf(A.size(1), A.size(0), A.buffer(), A.size(1), d_work.buffer(), ipiv.buffer(), m_gpu_info.buffer()), "Lapack call failed.");
                m_cpu_info = m_gpu_info;
                CALL_AND_RETHROW(cusolver::getrf_error_handling(m_cpu_info(0), 'a'));
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating LU decomposition.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate LU decomposition.");
            }
        }

        template <typename mat_type, typename mat_typeb, typename mat_typec>
        internal::valid_decomp_matrix_type_3<matrix_type, mat_type, mat_typeb, mat_typec, void>
        operator()(const mat_type &A, mat_typeb &L, mat_typec &U, vector<typename traits<backend_type>::int_type, backend_type> &ipiv)
        {
            try
            {
                RAISE_EXCEPTION("GPU Implementation of LU decomposition into distinct matrices is currently not supported.");
                CALL_AND_HANDLE(internal::lu_result_validation::validate_ipiv(A, ipiv), "Failed to validate pivot array.");

                bool is_wide = validate_L_U(A, L, U);
                if (is_wide)
                {
                    CALL_AND_HANDLE(U = A, "Failed to copy matrix.");
                    CALL_AND_HANDLE(blas_backend::getrf(U.size(1), U.size(0), U.buffer(), U.size(1), ipiv.buffer()), "Lapack call failed.");

                    // transfer the lower triangular part across to the L array.
                }
                else
                {
                    CALL_AND_HANDLE(L = A, "Failed to copy matrix.");
                    CALL_AND_HANDLE(blas_backend::getrf(L.size(1), L.size(0), L.buffer(), L.size(1), ipiv.buffer()), "Lapack call failed.");

                    // transfer the upper triangular part across to the U array.
                }
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating LU decomposition.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate LU decomposition.");
            }
        }
    };

} // namespace linalg

#endif // PYTTN_LINALG_DECOMPOSITIONS_LU_DECOMPOSITION_CUH_//
