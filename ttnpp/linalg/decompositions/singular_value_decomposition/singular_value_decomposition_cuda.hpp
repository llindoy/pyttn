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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_CUDA_HPP_
#define PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_CUDA_HPP_

#include "singular_value_decomposition_base.hpp"

#ifdef PYTTN_BUILD_CUDA

namespace linalg
{

    namespace internal
    {

#ifdef RSYEUDTHLRSIYUTHNRSYTUHRSIYTU
        template <typename T>
        struct singular_value_decomposition_helper<T, cuda_backend, true>
        {
            using int_type = cuda_backend::int_type;
            static_assert(is_number<T>::value, "Failed to initialise singular value decomposition working space object.");
            using size_type = cuda_backend::size_type;
            using memfill = memory::filler<T, cuda_backend>;

            struct additional_working
            {
                tensor<T, 1, cuda_backend> m_nref;
                tensor<int, 1, cuda_backend> m_gpu_info;
                tensor<int, 1> m_cpu_info;
                gesvdjInfo_t m_params;

                additional_working()
                {
                    m_gpu_info.resize(1);
                    m_cpu_info.resize(1);
                    m_nref.resize(1);
                    CALL_AND_HANDLE(cusolver_safe_call(cusolverDnCreateGesvdjInfo(&m_params)), "Failed to construct additional working object.  Failed to instantiate gesvdj parameters.");
                }
                ~additional_working() { cusolverDnDestroyGesvdjInfo(m_params); }

                void resize(size_type /* m */, size_type /* n */) {}
                void clear()
                {
                    CALL_AND_HANDLE(m_nref.clear(), "Failed to clear the nref array.");
                }
            };

            template <typename mat_type>
            static inline void call(bool compute_vectors, bool economy, mat_type &mat, T *S, T *U, const int_type LDU, T *VT, const int_type LDVT, T *WORK, const int_type LWORK, additional_working &working)
            {
                CALL_AND_RETHROW(call(compute_vectors, economy, mat.shape(0), mat.shape(1), mat.buffer(), mat.shape(1), S, U, LDU, VT, LDVT, WORK, LWORK, working));
            }

            static inline void call(bool compute_vectors, bool economy, const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *U, const int_type LDU, T *VT, const int_type LDVT, T *WORK, const int_type LWORK, additional_working &working)
            {
                // we need to compute A^T = VT^T S U^T as the lapack expects a column major matrix but we are passing in a row major matrix.
                cusolverEigMode_t JOBZ = (compute_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR);
                int_type econ = economy ? 1 : 0;
                CALL_AND_RETHROW(cuda_backend::gesvdj(JOBZ, econ, N, M, A, LDA, S, VT, LDVT, U, LDU, WORK, LWORK, working.m_gpu_info.buffer(), working.m_params));
                working.m_cpu_info = working.m_gpu_info;
                CALL_AND_RETHROW(cusolver::gesvd_error_handling(working.m_cpu_info(0), 'a'));
            }

            static inline int_type query_worksize(bool compute_vectors, bool economy, const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *U, const int_type LDU, T *VT, const int_type LDVT, additional_working &working)
            {
                cusolverEigMode_t JOBZ = (compute_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR);
                int_type econ = economy ? 1 : 0;
                int_type worksize;
                CALL_AND_RETHROW(cuda_backend::gesvdj_buffersize(JOBZ, econ, N, M, A, LDA, S, VT, LDVT, U, LDU, worksize, working.m_params));
                return worksize;
            }
        };

#endif

        template <typename matrix_type>
        class dense_matrix_singular_value_decomposition<matrix_type, false, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, cuda_backend>::value, void>::type>
        {
        public:
            using int_type = cuda_backend::int_type;
            using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
            using real_type = typename get_real_type<value_type>::type;
            using backend_type = typename traits<matrix_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using mem_trans = memory::transfer<backend_type, backend_type>;

        protected:
            tensor<value_type, 1, backend_type> m_work;
            tensor<value_type, 2, backend_type> m_mat;

            tensor<real_type, 1, cuda_backend> m_rwork;
            tensor<int, 1, cuda_backend> m_gpu_info;
            tensor<int, 1> m_cpu_info;
            tensor<value_type, 1, cuda_backend> m_nref;

        public:
            dense_matrix_singular_value_decomposition()
            {
                try
                {
                    CALL_AND_HANDLE(m_gpu_info.resize(1), "Failed to allocate gpu info buffer.");
                    CALL_AND_HANDLE(m_cpu_info.resize(1), "Failed to allocate cpu info buffer.");
                    CALL_AND_HANDLE(m_nref.resize(1), "Failed to allocate nref buffer.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct singular value decomposition object.");
                }
            }

            dense_matrix_singular_value_decomposition(size_type m, size_type n, bool requires_square_matrix = true)
            try : dense_matrix_singular_value_decomposition()
            {
                CALL_AND_HANDLE(resize(m, n, requires_square_matrix), "Failed to resize internal buffers.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct singular value decomposition object.");
            }

            template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
            dense_matrix_singular_value_decomposition(const mat_type &mat, bool requires_square_matrix = true)
            try : dense_matrix_singular_value_decomposition(mat.shape(0), mat.shape(1), requires_square_matrix) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct singular value decomposition object.");
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for resizing the internal buffers required for computing a singular values//
            // decomposition                                                                       //
            /////////////////////////////////////////////////////////////////////////////////////////
            void resize(size_type m, size_type n, bool requires_square_matrix = true)
            {
                try
                {
                    if (requires_square_matrix)
                    {
                        size_type maxmn = m < n ? n : m;
                        CALL_AND_HANDLE(m_mat.resize(maxmn, maxmn), "Failed to resize singular value decomposition engine object.  Failed when resizing internal matrix.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat.resize(m, n), "Failed to resize singular value decomposition engine object.  Failed when resizing internal matrix.");
                    }
                    size_type minmn = m < n ? m : n;
                    CALL_AND_HANDLE(m_rwork.resize(minmn - 1), "Failed to resize rwork array.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to resize singular value decomposition engine object.");
                }
            }

            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type &mat, bool requires_square_matrix = true)
            {
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), requires_square_matrix), "Failed to resize singular value decomposition object from a matrix.");
            }

            void resize_work_space(size_type worksize)
            {
                if (m_work.size() < worksize)
                {
                    std::cerr << worksize << std::endl;
                    CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of eigensolver object.");
                }
            }

            void clear()
            {
                try
                {
                    CALL_AND_HANDLE(m_mat.clear(), "Failed to clear the mat matrix.");
                    CALL_AND_HANDLE(m_work.clear(), "Failed to clear the work array.");
                    CALL_AND_HANDLE(m_rwork.clear(), "Failed to clear the rwork array.");
                    CALL_AND_HANDLE(m_nref.clear(), "Failed to clear the nref array.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to clear singular value decomposition engine object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for computing the singular values decomposition of a general dense matrix //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(const mat_type &mat, vals_type &S)
            {
                try
                {
                    CALL_AND_HANDLE(validate(mat, S), "The input matrices are invalid.");

                    size_type M = mat.shape(0);
                    size_type N = mat.shape(1);
                    // if N < M we need to transpose our input matrix and reorder the results
                    if (N < M)
                    {
                        N = mat.shape(0);
                        M = mat.shape(1);
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute matrix transpose necessary to form correct matrix shape for cusolver gesvd call.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix for cusolver gesvd call.");
                    }
                    size_type worksize;
                    CALL_AND_HANDLE(worksize = query_work_space(m_mat, S), "Failed to query the optimal worksize of the input array.");
                    CALL_AND_HANDLE(resize_work_space(worksize), "Failed tor resize workspace buffer.");

                    char jobu = 'N';
                    char jobvt = 'N';
                    value_type U;
                    value_type VT;
                    CALL_AND_HANDLE(
                        cuda_backend::gesvd(jobu, jobvt, N, M, m_mat.buffer(), m_mat.shape(1), S.buffer(), &VT, 1, &U, 1, m_work.buffer(), m_work.size(), m_rwork.buffer(), m_gpu_info.buffer()),
                        "Failed to evaluate cuda_backend gesvd call.");

                    CALL_AND_HANDLE(m_cpu_info = m_gpu_info, "Failed to copy status of cusolver gesvd call from device to host.");
                    CALL_AND_HANDLE(cusolver::gesvd_error_handling(m_cpu_info(0), 'a'), "Cusolver gesvd call failed.");
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("computing singular values of matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to compute singular values of matrix.");
                }
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> operator()(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true)
            {
                try
                {
                    char jobu, jobvt;
                    if (compute_full)
                    {
                        CALL_AND_RETHROW(validate(mat, S, U, Vh));
                        jobu = 'A';
                        jobvt = 'A';
                    }
                    else
                    {
                        CALL_AND_RETHROW(validate_minimal(mat, S, U, Vh));
                        jobu = 'S';
                        jobvt = 'S';
                    }

                    size_type M = mat.shape(0);
                    size_type N = mat.shape(1);
                    size_type MM = M;
                    size_type NN = N;
                    // if N < M we need to transpose our input matrix and reorder the results
                    if (NN < MM)
                    {
                        N = mat.shape(0);
                        M = mat.shape(1);
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute matrix transpose necessary to form correct matrix shape for cusolver gesvd call.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix for cusolver gesvd call.");
                    }
                    size_type worksize;
                    CALL_AND_HANDLE(worksize = query_work_space(m_mat, S), "Failed to query the optimal worksize of the input array.");
                    CALL_AND_HANDLE(resize_work_space(worksize), "Failed tor resize workspace buffer.");

                    // if we have transposed the input matrix we need to take that into account
                    if (NN < MM)
                    {
                        auto Ur = U.reinterpret_shape(U.shape(1), U.shape(0));
                        auto Vhr = Vh.reinterpret_shape(Vh.shape(1), Vh.shape(0));

                        CALL_AND_HANDLE(
                            cuda_backend::gesvd(jobu, jobvt, N, M, m_mat.buffer(), m_mat.shape(1), S.buffer(), U.buffer(), U.shape(0), Vh.buffer(), Vh.shape(0), m_work.buffer(), m_work.size(), m_rwork.buffer(), m_gpu_info.buffer()),
                            "Failed to evaluate cuda_backend gesvd call.");

                        CALL_AND_HANDLE(m_cpu_info = m_gpu_info, "Failed to copy status of cusolver gesvd call from device to host.");
                        CALL_AND_HANDLE(cusolver::gesvd_error_handling(m_cpu_info(0), 'a'), "Cusolver gesvd call failed.");

                        // if we are computing the full decomposition then the VT and U matrices are both square and so we can apply the
                        // resulting transformation inplace
                        CALL_AND_HANDLE(m_mat = trans(Ur), "Failed to transpose U matrix to get correct result.");
                        CALL_AND_HANDLE(U = m_mat, "Failed to assign the transposed U matrix.");
                        CALL_AND_HANDLE(m_mat = trans(Vhr), "Failed to transpose Vh matrix to get correct result.");
                        CALL_AND_HANDLE(Vh = m_mat, "Failed to assign the transposed Vh matrix.");
                    }
                    // otherwise we can just compute the svd
                    else
                    {
                        CALL_AND_HANDLE(
                            cuda_backend::gesvd(jobu, jobvt, N, M, m_mat.buffer(), m_mat.shape(1), S.buffer(), Vh.buffer(), Vh.shape(1), U.buffer(), U.shape(1), m_work.buffer(), m_work.size(), m_rwork.buffer(), m_gpu_info.buffer()),
                            "Failed to evaluate cuda_backend gesvd call.");

                        CALL_AND_HANDLE(m_cpu_info = m_gpu_info, "Failed to copy status of cusolver gesvd call from device to host.");
                        CALL_AND_HANDLE(cusolver::gesvd_error_handling(m_cpu_info(0), 'a'), "Cusolver gesvd call failed.");
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("computing singular value decomposition of matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to compute singular value decomposition of matrix.");
                }
            }

#ifdef RSITHUR
            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> operator()(mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true, bool keep_inputs = true)
            {
                CALL_AND_RETHROW(operator()(mat, S, U, Vh, compute_full));
            }

#endif
            //////////////////////////////////////////////////////////////////////////////////////////
            //  Functions for determining the optimal workspace for singular values decomposition   //
            //////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_type, vals_type> query_work_space(const mat_type &mat, vals_type &S)
            {
                int_type worksize;
                int_type M = mat.shape(0);
                int_type N = mat.shape(1);
                if (M <= N)
                {
                    CALL_AND_HANDLE(cuda_backend::gesvd_buffersize<value_type>(M, N, worksize), "Failed to determine optimal buffer size for cusolver gesvd call.");
                }
                else
                {
                    CALL_AND_HANDLE(cuda_backend::gesvd_buffersize<value_type>(N, M, worksize), "Failed to determine optimal buffer size for cusolver gesvd call.");
                }
                return static_cast<size_type>(worksize);
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> query_work_space(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true)
            {
                CALL_AND_HANDLE(return query_work_space(mat, S), "Failed to query workspace.");
            }

        protected:
            /////////////////////////////////////////////////////////////////////////////////////////
            //  Helper Functions for validating inputs and resizing the to singular values         //
            //  decomposition engine for a general matrix                                          //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            void validate(const mat_type &mat, vals_type &S)
            {
                CALL_AND_HANDLE(svd_result_validation::singular_values(mat, S), "Failed to singular values of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), false), "Failed to compute singular values of matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type>
            void validate_minimal(const mat_type &mat, vals_type &S)
            {
                CALL_AND_HANDLE(svd_result_validation::minimal_singular_values(mat, S), "Failed to singular values of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), false), "Failed to compute singular values of matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            void validate(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh)
            {
                CALL_AND_HANDLE(svd_result_validation::singular_values(mat, S), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(svd_result_validation::left_singular_vectors(mat, U), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the left singular vectors array.");
                CALL_AND_HANDLE(svd_result_validation::right_singular_vectors(mat, Vh), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the right singular vvectors array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), true), "Failed to compute singular value decomposition of matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            void validate_minimal(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh)
            {
                CALL_AND_HANDLE(svd_result_validation::minimal_singular_values(mat, S), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(svd_result_validation::minimal_left_singular_vectors(mat, U), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the left singular vectors array.");
                CALL_AND_HANDLE(svd_result_validation::minimal_right_singular_vectors(mat, Vh), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the right singular vectors array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), false), "Failed to compute singular value decomposition of matrix.  Failed resize buffers.");
            }
        };

    } // namespace internal

} // namespace linalg

#endif // PYTTN_BUILD_CUDA

#endif // PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_CUDA_HPP_
