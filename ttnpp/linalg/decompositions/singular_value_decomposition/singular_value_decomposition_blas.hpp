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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_BLAS_HPP_
#define PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_BLAS_HPP_

#include "singular_value_decomposition_base.hpp"

namespace linalg
{

    namespace internal
    {

        template <typename T>
        struct singular_value_decomposition_helper<T, blas_backend, false>
        {
            using int_type = blas_backend::int_type;
            static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to initialise singular value decomposition working space object.");
            using size_type = blas_backend::size_type;
            using memfill = memory::filler<T, blas_backend>;

            struct additional_working
            {
                void resize(size_type /* m */, size_type /* n */) {}
                void clear() {}
            };

            static inline void call(bool compute_vectors, const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *U, const int_type LDU, T *VT, const int_type LDVT, T *WORK, const int_type LWORK, additional_working & /* working */)
            {
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesvd(JOBZ, JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, WORK, LWORK));
            }

            static inline int_type query_worksize(bool compute_vectors, const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *U, const int_type LDU, T *VT, const int_type LDVT, additional_working & /* working */)
            {
                T worksize;
                int_type lwork = -1;
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesvd(JOBZ, JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, &worksize, lwork));
                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }

            static inline void call_inplace(const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *R, const int_type LDR, T *WORK, const int_type LWORK, additional_working & /* working */)
            {
                int_type MM = N;
                int_type NN = M;
                T nref;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('A', 'O', MM, NN, A, LDA, S, R, LDR, &nref, 1, WORK, LWORK));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('O', 'A', MM, NN, A, LDA, S, &nref, 1, R, LDR, WORK, LWORK));
                }
            }

            static inline int_type query_worksize_inplace(const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *R, const int_type LDR, additional_working & /* working */)
            {
                T worksize, nref;
                int_type lwork = -1;
                int_type MM = N;
                int_type NN = M;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('A', 'O', MM, NN, A, LDA, S, R, LDR, &nref, 1, &worksize, lwork));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('O', 'A', MM, NN, A, LDA, S, &nref, 1, R, LDR, &worksize, lwork));
                }

                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }
        };

        template <typename T>
        struct singular_value_decomposition_helper<complex<T>, blas_backend, false>
        {
            using int_type = blas_backend::int_type;
            static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to initialise singular value decomposition working space object.");
            using size_type = blas_backend::size_type;
            using memfill = memory::filler<complex<T>, blas_backend>;

            struct additional_working
            {
                tensor<T, 1> m_rwork;
                void resize(size_type m, size_type n)
                {
                    size_type mn = m < n ? m : n;
                    CALL_AND_RETHROW(m_rwork.resize(5 * mn));
                }
                void clear() { CALL_AND_RETHROW(m_rwork.clear()); }
            };
            static inline void call(bool compute_vectors, const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *U, const int_type LDU, complex<T> *VT, const int_type LDVT, complex<T> *WORK, const int_type LWORK, additional_working &working)
            {
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesvd(JOBZ, JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, WORK, LWORK, working.m_rwork.buffer()));
            }

            static inline int_type query_worksize(bool compute_vectors, const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *U, const int_type LDU, complex<T> *VT, const int_type LDVT, additional_working &working)
            {
                complex<T> worksize;
                int_type lwork = -1;
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesvd(JOBZ, JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, &worksize, lwork, working.m_rwork.buffer()));
                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }

            static inline void call_inplace(const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *R, const int_type LDR, complex<T> *WORK, const int_type LWORK, additional_working &working)
            {
                int_type MM = N;
                int_type NN = M;
                complex<T> nref;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('A', 'O', MM, NN, A, LDA, S, R, LDR, &nref, 1, WORK, LWORK, working.m_rwork.buffer()));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('O', 'A', MM, NN, A, LDA, S, &nref, 1, R, LDR, WORK, LWORK, working.m_rwork.buffer()));
                }
            }

            static inline int_type query_worksize_inplace(const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *R, const int_type LDR, additional_working &working)
            {
                complex<T> worksize, nref;
                int_type lwork = -1;
                int_type MM = N;
                int_type NN = M;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('A', 'O', MM, NN, A, LDA, S, R, LDR, &nref, 1, &worksize, lwork, working.m_rwork.buffer()));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesvd('O', 'A', MM, NN, A, LDA, S, &nref, 1, R, LDR, &worksize, lwork, working.m_rwork.buffer()));
                }
                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }
        };

        template <typename T>
        struct singular_value_decomposition_helper<T, blas_backend, true>
        {
            using int_type = blas_backend::int_type;
            static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to initialise singular value decomposition working space object.");
            using size_type = blas_backend::size_type;
            using memfill = memory::filler<T, blas_backend>;

            struct additional_working
            {
                linalg::tensor<int, 1, blas_backend> m_iwork;
                void resize(size_type m, size_type n)
                {
                    size_type mn = m < n ? m : n;
                    CALL_AND_RETHROW(m_iwork.resize(8 * mn));
                }
                void clear() { CALL_AND_RETHROW(m_iwork.clear()); }
            };

            static inline void call(bool compute_vectors, const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *U, const int_type LDU, T *VT, const int_type LDVT, T *WORK, const int_type LWORK, additional_working &working)
            {
                // we need to compute A^T = VT^T S U^T as lapack expects a column major matrix but we are passing in a row major matrix.
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesdd(JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, WORK, LWORK, working.m_iwork.buffer()));
            }

            static inline int_type query_worksize(bool compute_vectors, const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *U, const int_type LDU, T *VT, const int_type LDVT, additional_working &working)
            {
                T worksize;
                int_type lwork = -1;
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesdd(JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, &worksize, lwork, working.m_iwork.buffer()));
                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }

            static inline void call_inplace(const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *R, const int_type LDR, T *WORK, const int_type LWORK, additional_working &working)
            {
                int_type MM = N;
                int_type NN = M;
                T nref;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, R, LDR, &nref, 1, WORK, LWORK, working.m_iwork.buffer()));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, &nref, 1, R, LDR, WORK, LWORK, working.m_iwork.buffer()));
                }
            }

            static inline int_type query_worksize_inplace(const int_type M, const int_type N, T *A, const int_type LDA, T *S, T *R, const int_type LDR, additional_working &working)
            {
                T worksize, nref;
                int_type lwork = -1;
                int_type MM = N;
                int_type NN = M;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, R, LDR, &nref, 1, &worksize, lwork, working.m_iwork.buffer()));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, &nref, 1, R, LDR, &worksize, lwork, working.m_iwork.buffer()));
                }

                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }
        };

        template <typename T>
        struct singular_value_decomposition_helper<complex<T>, blas_backend, true>
        {
            using int_type = blas_backend::int_type;
            static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to initialise singular value decomposition working space object.");
            using size_type = blas_backend::size_type;
            using memfill = memory::filler<complex<T>, blas_backend>;

            struct additional_working
            {
                tensor<T, 1> m_rwork;
                tensor<int, 1> m_iwork;
                void resize(size_type m, size_type n)
                {
                    size_type mn = m < n ? m : n;
                    size_type rworksize = 5 * mn;
                    // resize for the divide and conquer algorithm if we are using it
                    size_type mx = m < n ? n : m;
                    size_type r1 = 5 * mn * mn + 5 * mn;
                    size_type r2 = 2 * mx * mn + 2 * mn * mn + mn;

                    size_type r = r1 < r2 ? r2 : r1;
                    rworksize = rworksize < r ? r : rworksize;
                    CALL_AND_RETHROW(m_iwork.resize(8 * mn));
                    CALL_AND_RETHROW(m_rwork.resize(rworksize));
                }

                void clear()
                {
                    CALL_AND_RETHROW(m_rwork.clear());
                    CALL_AND_RETHROW(m_iwork.clear());
                }
            };
            static inline void call(bool compute_vectors, const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *U, const int_type LDU, complex<T> *VT, const int_type LDVT, complex<T> *WORK, const int_type LWORK, additional_working &working)
            {
                // we need to compute A^T = VT^T S U^T as lapack expects a column major matrix but we are passing in a row major matrix.
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesdd(JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, WORK, LWORK, working.m_rwork.buffer(), working.m_iwork.buffer()));
            }

            static inline int_type query_worksize(bool compute_vectors, const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *U, const int_type LDU, complex<T> *VT, const int_type LDVT, additional_working &working)
            {
                complex<T> worksize;
                int_type lwork = -1;
                char JOBZ = (compute_vectors ? 'A' : 'N');
                CALL_AND_RETHROW(blas_backend::gesdd(JOBZ, N, M, A, LDA, S, VT, LDVT, U, LDU, &worksize, lwork, working.m_rwork.buffer(), working.m_iwork.buffer()));
                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }

            static inline void call_inplace(const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *R, const int_type LDR, complex<T> *WORK, const int_type LWORK, additional_working &working)
            {
                int_type MM = N;
                int_type NN = M;
                complex<T> nref;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, R, LDR, &nref, 1, WORK, LWORK, working.m_rwork.buffer(), working.m_iwork.buffer()));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, &nref, 1, R, LDR, WORK, LWORK, working.m_rwork.buffer(), working.m_iwork.buffer()));
                }
            }

            static inline int_type query_worksize_inplace(const int_type M, const int_type N, complex<T> *A, const int_type LDA, T *S, complex<T> *R, const int_type LDR, additional_working &working)
            {
                complex<T> worksize, nref;
                int_type lwork = -1;
                int_type MM = N;
                int_type NN = M;
                // if MM < NN then the second possible result argument (U^T) is not referenced so R must store (VT^T) and A will store U^T on exit
                if (MM < NN)
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, R, LDR, &nref, 1, &worksize, lwork, working.m_rwork.buffer(), working.m_iwork.buffer()));
                }
                // otherwise the first result argument (VT^T) is not referenced
                else
                {
                    CALL_AND_RETHROW(blas_backend::gesdd('O', MM, NN, A, LDA, S, &nref, 1, R, LDR, &worksize, lwork, working.m_rwork.buffer(), working.m_iwork.buffer()));
                }
                CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
            }
        };

        template <typename matrix_type, bool use_divide_and_conquer>
        class dense_matrix_singular_value_decomposition<matrix_type, use_divide_and_conquer, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, blas_backend>::value, void>::type>
        {
        public:
            using int_type = blas_backend::int_type;
            using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
            using backend_type = typename traits<matrix_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using mem_trans = memory::transfer<backend_type, backend_type>;
            using helper = singular_value_decomposition_helper<value_type, backend_type, use_divide_and_conquer>;

        protected:
            typename helper::additional_working m_additional;
            tensor<value_type, 1, backend_type> m_work;
            tensor<value_type, 2, backend_type> m_mat;

        public:
            dense_matrix_singular_value_decomposition() {}
            dense_matrix_singular_value_decomposition(size_type m, size_type n, bool requires_matrix = true) { CALL_AND_HANDLE(resize(m, n, requires_matrix), "Failed construct singular value decomposition engine with preallocated size.  Failed when resizing buffers."); }
            template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
            dense_matrix_singular_value_decomposition(const mat_type &mat, bool requires_matrix = true) { CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), requires_matrix), "Failed construct singular value decomposition engine with preallocated size.  Failed when resizing buffers."); }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for resizing the internal buffers required for computing a singular values//
            // decomposition                                                                       //
            /////////////////////////////////////////////////////////////////////////////////////////
            void resize(size_type m, size_type n, bool /*requires_matrix*/ = true)
            {
                try
                {
                    CALL_AND_HANDLE(m_mat.resize(m, n), "Failed when resizing internal matrix.");
                    CALL_AND_HANDLE(m_additional.resize(m, n), "Failed when resizing additional working space.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to resize singular value decomposition object.");
                }
            }
            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type &mat, bool requires_matrix = true) { CALL_AND_RETHROW(resize(mat.shape(0), mat.shape(1), requires_matrix)); }
            void resize_work_space(size_type worksize)
            {
                if (m_work.size() < worksize)
                {
                    CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of eigensolver object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for clearing the single value decomposition engine
            /////////////////////////////////////////////////////////////////////////////////////////
            void clear()
            {
                try
                {
                    CALL_AND_HANDLE(m_mat.clear(), "Failed to clear the mat matrix.");
                    CALL_AND_HANDLE(m_work.clear(), "Failed to clear the work array.");
                    CALL_AND_HANDLE(m_additional.clear(), "Failed to clear the additional working array.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to clear singular value decomposition object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for computing the singular values decomposition of a general dense matrix //
            /////////////////////////////////////////////////////////////////////////////////////////

            // Functions for just computing the singular values
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(const mat_type &mat, vals_type &S)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, S, true));
                    CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix into working buffer.");
                    CALL_AND_RETHROW(compute(m_mat, S));
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating singular values.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate singular values.");
                }
            }

            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(mat_type &mat, vals_type &S, bool keep_inputs = false)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, S, true));
                    if (keep_inputs)
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix into working buffer.");
                        CALL_AND_RETHROW(compute(m_mat, S));
                    }
                    else
                    {
                        CALL_AND_RETHROW(compute(mat, S));
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating singular values.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate singular values.");
                }
            }

            // Function for computing the singular values inplace.  This can't take a constant vector as it always overwrites the mat matrix.
            template <typename mat_type, typename vals_type, typename rvecs_type>
            valid_mutable_decomp_func<matrix_type, mat_type, svd_result_ordering, validate_real_vals_vecs_type, vals_type, rvecs_type> operator()(mat_type &mat, vals_type &S, rvecs_type &R)
            {
                try
                {
                    CALL_AND_HANDLE(svd_result_validation::minimal_singular_values(mat, S), "Failed to ensure the correct shape for the singular values array.");
                    size_type MM = mat.shape(1);
                    size_type NN = mat.shape(0);
                    if (MM < NN)
                    {
                        CALL_AND_HANDLE(svd_result_validation::minimal_right_singular_vectors(mat, R), "Failed to ensure the correct shape for the right singular vector matrix.");
                        CALL_AND_RETHROW(compute_inplace(mat, S, R));
                        return svd_result_ordering::usv;
                    }
                    else
                    {
                        CALL_AND_HANDLE(svd_result_validation::minimal_left_singular_vectors(mat, R), "Failed to ensure the correct shape for the left singular vector matrix.");
                        CALL_AND_RETHROW(compute_inplace(mat, S, R));
                        return svd_result_ordering::vsu;
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating singular value decomposition.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate singular value decomposition.");
                }
            }

            // Functions for computing the singular values and left and right singular vectors
            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> operator()(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true)
            {
                try
                {
                    if (compute_full)
                    {
                        CALL_AND_RETHROW(validate(mat, S, U, Vh, true));
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix into working buffer.");
                        CALL_AND_RETHROW(compute(m_mat, S, U, Vh));
                    }
                    else
                    {
                        CALL_AND_RETHROW(validate_minimal(mat, S, U, Vh, false));
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix into working buffer.");
                        CALL_AND_RETHROW(compute_minimal(m_mat, S, U, Vh));
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating singular value decomposition.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate singular value decomposition.");
                }
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> operator()(mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true, bool keep_inputs = false)
            {
                try
                {
                    if (compute_full)
                    {
                        CALL_AND_RETHROW(validate(mat, S, U, Vh, true));
                        if (keep_inputs)
                        {
                            CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix into working buffer.");
                            CALL_AND_RETHROW(compute(m_mat, S, U, Vh));
                        }
                        else
                        {
                            CALL_AND_RETHROW(compute(mat, S, U, Vh));
                        }
                    }
                    else
                    {
                        CALL_AND_RETHROW(validate_minimal(mat, S, U, Vh, false));
                        if (keep_inputs)
                        {
                            CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix into working buffer.");
                            CALL_AND_RETHROW(compute_minimal(m_mat, S, U, Vh));
                        }
                        else
                        {
                            CALL_AND_RETHROW(compute_minimal(mat, S, U, Vh));
                        }
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating singular value decomposition.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate singular value decomposition.");
                }
            }

            //////////////////////////////////////////////////////////////////////////////////////////
            //  Functions for determining the optimal workspace for singular valuesdecomposition    //
            //////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_type, vals_type> query_work_space(const mat_type &mat, vals_type &S)
            {
                value_type U, V;
                CALL_AND_HANDLE(m_mat = mat, "Failed to compute squery optimal workspace for singular value decomposition.  Failed to copy input matrix into working buffer.");
                CALL_AND_HANDLE(return helper::query_worksize(false, mat.shape(0), mat.shape(1), m_mat.buffer(), m_mat.shape(1), S.buffer(), &U, 1, &V, 1, m_additional), "Failed to query optimal workspace for the singular value decomposition.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> query_work_space(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true)
            {
                if (compute_full)
                {
                    CALL_AND_HANDLE(m_mat = mat, "Failed to compute squery optimal workspace for singular value decomposition.  Failed to copy input matrix into working buffer.");
                    CALL_AND_HANDLE(return helper::query_worksize(true, mat.shape(0), mat.shape(1), m_mat.buffer(), mat.shape(1), S.buffer(), U.buffer(), U.shape(1), Vh.buffer(), Vh.shape(1), m_additional), "Failed to query optimal workspace for the singular value decomposition.");
                }
                else
                {
                    size_type MM = mat.shape(1);
                    size_type NN = mat.shape(0);
                    if (MM < NN)
                    {
                        CALL_AND_RETHROW(return query_work_space(U, S, Vh));
                    }
                    else
                    {
                        CALL_AND_RETHROW(return query_work_space(Vh, S, U));
                    }
                }
            }

            template <typename mat_type, typename vals_type, typename rvecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_vecs_type, vals_type, rvecs_type> query_work_space(const mat_type &mat, vals_type &S, rvecs_type &r)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to compute squery optimal workspace for singular value decomposition.  Failed to copy input matrix into working buffer.");
                CALL_AND_HANDLE(return helper::query_worksize_inplace(mat.shape(0), mat.shape(1), m_mat.buffer(), mat.shape(1), S.buffer(), r.buffer(), r.shape(1), m_additional), "Failed to query the optimal workspace for the singular value decomposition.");
            }

        protected:
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_type, vals_type> query_work_space_(mat_type &mat, vals_type &S)
            {
                value_type U, V;
                CALL_AND_HANDLE(return helper::query_worksize(false, mat.shape(0), mat.shape(1), mat.buffer(), mat.shape(1), S.buffer(), &U, 1, &V, 1, m_additional), "Failed to query optimal workspace for the singular value decomposition.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> query_work_space_(mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true)
            {
                if (compute_full)
                {
                    CALL_AND_HANDLE(return helper::query_worksize(true, mat.shape(0), mat.shape(1), mat.buffer(), mat.shape(1), S.buffer(), U.buffer(), U.shape(1), Vh.buffer(), Vh.shape(1), m_additional), "Failed to query optimal workspace for the singular value decomposition.");
                }
                else
                {
                    size_type MM = mat.shape(1);
                    size_type NN = mat.shape(0);
                    if (MM < NN)
                    {
                        CALL_AND_RETHROW(return query_work_space(U, S, Vh));
                    }
                    else
                    {
                        CALL_AND_RETHROW(return query_work_space(Vh, S, U));
                    }
                }
            }

            template <typename mat_type, typename vals_type, typename rvecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_vecs_type, vals_type, rvecs_type> query_work_space_(mat_type &mat, vals_type &S, rvecs_type &r)
            {
                CALL_AND_HANDLE(return helper::query_worksize_inplace(mat.shape(0), mat.shape(1), mat.buffer(), mat.shape(1), S.buffer(), r.buffer(), r.shape(1), m_additional), "Failed to query the optimal workspace for the singular value decomposition.");
            }

        protected:
            /////////////////////////////////////////////////////////////////////////////////////////
            //  Helper Functions for validating inputs and resizing the to singular values         //
            //  decomposition engine for a general matrix                                          //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            void validate(const mat_type &mat, vals_type &S, bool use_matrix)
            {
                CALL_AND_HANDLE(svd_result_validation::singular_values(mat, S), "Failed to singular values of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), use_matrix), "Failed to compute singular values of matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type>
            void validate_minimal(const mat_type &mat, vals_type &S, bool use_matrix)
            {
                CALL_AND_HANDLE(svd_result_validation::minimal_singular_values(mat, S), "Failed to singular values of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), use_matrix), "Failed to compute singular values of matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            void validate(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool use_matrix)
            {
                CALL_AND_HANDLE(svd_result_validation::singular_values(mat, S), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(svd_result_validation::left_singular_vectors(mat, U), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the left singular vectors array.");
                CALL_AND_HANDLE(svd_result_validation::right_singular_vectors(mat, Vh), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the right singular vvectors array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), use_matrix), "Failed to compute singular value decomposition of matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            void validate_minimal(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool use_matrix)
            {
                CALL_AND_HANDLE(svd_result_validation::minimal_singular_values(mat, S), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the singular values array.");
                CALL_AND_HANDLE(svd_result_validation::minimal_left_singular_vectors(mat, U), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the left singular vectors array.");
                CALL_AND_HANDLE(svd_result_validation::minimal_right_singular_vectors(mat, Vh), "Failed to singular value decomposition of matrix.  Failed to ensure the correct shape for the right singular vectors array.");
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), use_matrix), "Failed to compute singular value decomposition of matrix.  Failed resize buffers.");
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            //  Helper Functions for computing the singular value decomposition of a general matrix//
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            void compute(mat_type &mat, vals_type &S)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space_(mat, S), "Failed to compute singular values matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_work_space(worksize), "Failed to compute singular values of matrix. Failed to resize the optimal workspace array.");

                value_type U;
                value_type V;
                CALL_AND_HANDLE(helper::call(false, mat.shape(0), mat.shape(1), mat.buffer(), mat.shape(1), S.buffer(), &U, 1, &V, 1, m_work.buffer(), m_work.size(), m_additional), "Failed to compute singular values of matrix.  Failed when evaluating the svd.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            void compute(mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space_(mat, S, U, Vh), "Failed to compute singular values matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_work_space(worksize), "Failed to compute singular value decomposition of matrix. Failed to resize the optimal workspace array.");
                CALL_AND_HANDLE(helper::call(true, mat.shape(0), mat.shape(1), mat.buffer(), mat.shape(1), S.buffer(), U.buffer(), U.shape(1), Vh.buffer(), Vh.shape(1), m_work.buffer(), m_work.size(), m_additional), "Failed to compute singular value decomposition of matrix.  Failed when evaluating the svd.");
            }

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            void compute_minimal(mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh)
            {
                // now we determine which of the two inplace transforms we can perform and copy the matrix values into the require U or Vh array
                size_type MM = mat.shape(1);
                size_type NN = mat.shape(0);
                if (MM < NN)
                {
                    CALL_AND_HANDLE(U = mat, "Failed to compute singular value decomposition of matrix.  Failed to copy input matrix into working buffer.");
                    CALL_AND_RETHROW(compute_inplace(U, S, Vh));
                }
                else
                {
                    CALL_AND_HANDLE(Vh = mat, "Failed to compute singular value decomposition of matrix.  Failed to copy input matrix into working buffer.")
                    CALL_AND_RETHROW(compute_inplace(Vh, S, U));
                }
            }

            template <typename mat_type, typename vals_type, typename rvecs_type>
            void compute_inplace(mat_type &U, vals_type &S, rvecs_type &Vh)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space_(U, S, Vh), "Failed to compute singular values matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_work_space(worksize), "Failed to compute singular value decomposition of matrix. Failed to resize the optimal workspace array.");
                CALL_AND_HANDLE(helper::call_inplace(U.shape(0), U.shape(1), U.buffer(), U.shape(1), S.buffer(), Vh.buffer(), Vh.shape(1), m_work.buffer(), m_work.size(), m_additional), "Failed to compute singular value decomposition of matrix.   Failed to evaluate svd.");
            }
        };
    } // namespace internal

} // namespace linalg

#endif // PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_BLAS_HPP_
