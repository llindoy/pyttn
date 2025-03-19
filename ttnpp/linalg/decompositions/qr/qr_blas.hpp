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

 #ifndef PYTTN_LINALG_DECOMPOSITIONS_QR_BLAS_HPP_
 #define PYTTN_LINALG_DECOMPOSITIONS_QR_BLAS_HPP_

#include "qr_base.hpp"

namespace linalg
{

    namespace internal
    {
        template <typename T>
        struct qr_helper<T, blas_backend>
        {
            using int_type = blas_backend::int_type;
            static_assert(is_number<T>::value && !is_complex<T>::value,
                          "Failed to initialise singular value decomposition working "
                          "space object.");
            using size_type = blas_backend::size_type;
            using memfill = memory::filler<T, blas_backend>;

            struct additional_working
            {
                void resize(size_type /* m */, size_type /* n */) {}
                void clear() {}
            };

            static inline void call_lq(const int_type m, const int_type n, T *a,
                                       const int_type lda, T *tau, T *work,
                                       const int_type lwork)
            {
                CALL_AND_HANDLE(blas_backend::gelqf(m, n, a, lda, tau, work, lwork),
                                "Failed to perform lq call.");
            }

            static inline void call_qr(const int_type m, const int_type n, T *a,
                                       const int_type lda, T *tau, T *work,
                                       const int_type lwork)
            {
                CALL_AND_HANDLE(blas_backend::geqrf(m, n, a, lda, tau, work, lwork),
                                "Failed to perform qr call.");
            }

            static inline void call_qr(const int_type m, const int_type n, T *a,
                                       const int_type lda, int_type *jpvt, T *tau,
                                       T *work, const int_type lwork,
                                       additional_working & /* working */)
            {
                CALL_AND_HANDLE(blas_backend::geqp3(m, n, a, lda, jpvt, tau, work, lwork),
                                "Failed to perform qr call.");
            }
        };

        template <typename T>
        struct qr_helper<complex<T>, blas_backend>
        {
            using int_type = blas_backend::int_type;
            static_assert(is_number<T>::value && !is_complex<T>::value,
                          "Failed to initialise singular value decomposition working "
                          "space object.");
            using size_type = blas_backend::size_type;
            using memfill = memory::filler<T, blas_backend>;

            struct additional_working
            {
                tensor<T, 1> m_rwork;
                void resize(size_type m, size_type n)
                {
                    CALL_AND_RETHROW(m_rwork.resize(2 * std::max(m, n)));
                }
                void clear() { CALL_AND_RETHROW(m_rwork.clear()); }
            };

            static inline void call_lq(const int_type m, const int_type n, complex<T> *a,
                                       const int_type lda, complex<T> *tau,
                                       complex<T> *work, const int_type lwork)
            {
                CALL_AND_HANDLE(blas_backend::gelqf(m, n, a, lda, tau, work, lwork),
                                "Failed to perform lq call.");
            }

            static inline void call_qr(const int_type m, const int_type n, complex<T> *a,
                                       const int_type lda, complex<T> *tau,
                                       complex<T> *work, const int_type lwork)
            {
                CALL_AND_HANDLE(blas_backend::geqrf(m, n, a, lda, tau, work, lwork),
                                "Failed to perform qr call.");
            }

            static inline void call_qr(const int_type m, const int_type n, complex<T> *a,
                                       const int_type lda, int_type *jpvt,
                                       complex<T> *tau, complex<T> *work,
                                       const int_type lwork,
                                       additional_working &working)
            {
                CALL_AND_HANDLE(blas_backend::geqp3(m, n, a, lda, jpvt, tau, work, lwork,
                                                    working.m_rwork.buffer()),
                                "Failed to perform qr call.");
            }
        };

        // class for computing the QR decomposition of a dense row-major order matrix,
        // A.  As lapack expects column major matrices, Lapack will actually be working
        // with A^T.  We could avoid this by transposing the array, but here instead we
        // simply do an LQ decomposition of A^T as this is equivalent to a QR
        // decomposition of A
        template <typename matrix_type>
        class dense_matrix_qr<
            matrix_type,
            typename std::enable_if<
                is_dense_matrix<matrix_type>::value &&
                    std::is_same<typename traits<matrix_type>::backend_type,
                                 blas_backend>::value,
                void>::type>
        {
        public:
            using int_type = blas_backend::int_type;
            using value_type =
                typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
            using backend_type = typename traits<matrix_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using mem_trans = memory::transfer<backend_type, backend_type>;

            using helper = qr_helper<value_type, backend_type>;

        protected:
            tensor<value_type, 1, backend_type> m_tau;
            tensor<value_type, 1, backend_type> m_work;
            tensor<value_type, 2, backend_type> m_mat;

        public:
            dense_matrix_qr() {}
            dense_matrix_qr(size_type m, size_type n, bool requires_matrix = true)
            {
                CALL_AND_HANDLE(resize(m, n, requires_matrix),
                                "Failed construct singular value decomposition engine with "
                                "preallocated size.  Failed when resizing buffers.");
            }
            template <typename mat_type,
                      typename =
                          typename std::enable_if<is_tensor<mat_type>::value, void>::type,
                      typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
            dense_matrix_qr(const mat_type &mat, bool requires_matrix = true)
            {
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), requires_matrix),
                                "Failed construct singular value decomposition engine with "
                                "preallocated size.  Failed when resizing buffers.");
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for resizing the internal buffers required for computing a qr //
            // decomposition //
            /////////////////////////////////////////////////////////////////////////////////////////
            void resize(size_type m, size_type n, bool requires_matrix = true)
            {
                try
                {
                    CALL_AND_HANDLE(m_tau.resize(m > n ? n : m),
                                    "Failed when resizing internal matrix.");
                    if (requires_matrix)
                    {
                        CALL_AND_HANDLE(m_mat.resize(m, n),
                                        "Failed when resizing internal matrix.");
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to resize singular value decomposition object.");
                }
            }
            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, void>
            resize(const mat_type &mat, bool requires_matrix = true)
            {
                CALL_AND_RETHROW(resize(mat.shape(0), mat.shape(1), requires_matrix));
            }
            void resize_work_space(size_type worksize)
            {
                if (m_work.size() < worksize)
                {
                    CALL_AND_HANDLE(m_work.resize(worksize),
                                    "Failed to resize workspace of eigensolver object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for clearing the qr decomposition engine
            /////////////////////////////////////////////////////////////////////////////////////////
            void clear()
            {
                try
                {
                    CALL_AND_HANDLE(m_work.clear(), "Failed to clear the work array.");
                    CALL_AND_HANDLE(m_tau.clear(),
                                    "Failed to clear the additional working array.");
                    CALL_AND_HANDLE(m_mat.clear(),
                                    "Failed to clear the additional working array.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to clear singular value decomposition object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for computing the qr decomposition of a general dense matrix //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename mat_typeb, typename mat_typec>
            valid_decomp_matrix_type_3<matrix_type, mat_type, mat_typeb, mat_typec, void>
            operator()(const mat_type &mat, mat_typeb &Q, mat_typec &R)
            {
                try
                {
                    CALL_AND_HANDLE(resize(mat, true), "Failed to resize internal buffers.");
                    CALL_AND_HANDLE(m_mat = mat,
                                    "Failed to copy input matrix into working buffer.");
                    CALL_AND_RETHROW(compute(m_mat, Q, R));
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate qr.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for computing the qr decomposition of a general dense matrix //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename mat_typeb, typename mat_typec>
            valid_decomp_matrix_type_3<matrix_type, mat_type, mat_typeb, mat_typec, void>
            operator()(mat_type &mat, mat_typeb &Q, mat_typec &R,
                       bool keep_matrix = true)
            {
                try
                {
                    CALL_AND_HANDLE(resize(mat, !keep_matrix),
                                    "Failed to resize internal buffers.");
                    if (keep_matrix)
                    {
                        CALL_AND_HANDLE(m_mat = mat,
                                        "Failed to copy input matrix into working buffer.");
                        CALL_AND_RETHROW(compute(m_mat, Q, R));
                    }
                    else
                    {
                        CALL_AND_RETHROW(compute(mat, Q, R));
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate qr.");
                }
            }

        protected:
            template <typename mat_type, typename mat_typeb, typename mat_typec>
            void compute(mat_type &mat, mat_typeb &Q, mat_typec &R)
            {
                try
                {
                    size_type rank = std::min(mat.shape(0), mat.shape(1));

                    value_type worksize;
                    int_type lwork = -1;
                    // ensure that the worksize buffer is the correct size.
                    CALL_AND_HANDLE(helper::call_lq(mat.shape(1), mat.shape(0), mat.buffer(),
                                                    mat.shape(1), m_tau.buffer(), &worksize,
                                                    lwork),
                                    "Failed to query worksize for QR decomposition");
                    size_type iworksize = internal::worksize_as_integer(worksize);
                    CALL_AND_HANDLE(resize_work_space(iworksize),
                                    "Failed to compute singular value decomposition of "
                                    "matrix. Failed to resize the optimal workspace array.");

                    // now actually perform the LQ decompositions of A as a column major
                    // matrix (e.g. perform LQ of A^T)
                    CALL_AND_HANDLE(helper::call_lq(mat.shape(1), mat.shape(0), mat.buffer(),
                                                    mat.shape(1), m_tau.buffer(),
                                                    m_work.buffer(), m_work.size()),
                                    "Failed to compute R matrix and the elementary "
                                    "reflectors defining the unitary.");

                    // now copy the matrix into the correct location
                    CALL_AND_HANDLE(R.resize(rank, mat.shape(1)),
                                    "Failed to resize the R buffer.");
                    for (size_type row = 0; row < rank; ++row)
                    {
                        std::fill_n(R.buffer() + row * mat.shape(1), row, value_type(0));
                        std::copy_n(mat.buffer() + row * (mat.shape(1) + 1), mat.shape(1) - row,
                                    R.buffer() + row * (mat.shape(1) + 1));
                    }

                    // now resize the workbuffer for constructing the unitary
                    CALL_AND_HANDLE(blas_backend::unglq(rank, mat.shape(0), rank,
                                                        mat.buffer(), mat.shape(1),
                                                        m_tau.buffer(), &worksize, lwork),
                                    "Failed to query worksize for constructing unitary from "
                                    "elementary reflectors.");
                    iworksize = internal::worksize_as_integer(worksize);
                    CALL_AND_HANDLE(resize_work_space(iworksize),
                                    "Failed to compute singular value decomposition of "
                                    "matrix. Failed to resize the optimal workspace array.");
                    CALL_AND_HANDLE(blas_backend::unglq(
                                        rank, mat.shape(0), rank, mat.buffer(), mat.shape(1),
                                        m_tau.buffer(), m_work.buffer(), m_work.size()),
                                    "Failed to compute unitary from elementary reflectors.");

                    CALL_AND_HANDLE(Q.resize(mat.shape(0), rank),
                                    "Failed to resize the unitary buffer.");
                    if (mat.shape(0) == mat.shape(1))
                    {
                        CALL_AND_HANDLE(Q = mat, "Failed to copy matrix into Q array.");
                    }
                    else
                    {
                        for (size_type row = 0; row < mat.shape(0); ++row)
                        {
                            std::copy_n(mat.buffer() + row * mat.shape(1), rank,
                                        Q.buffer() + row * rank);
                        }
                    }
                }
                catch (const invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating qr.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate qr.");
                }
            }
        };

        // class for computing the LQ decomposition of a dense row-major order matrix,
        // A.  As lapack expects column major matrices, Lapack will actually be working
        // with A^T.  We could avoid this by transposing the array, but here instead we
        // simply do an LQ decomposition of A^T as this is equivalent to a LQ
        // decomposition of A
        template <typename matrix_type>
        class dense_matrix_lq<
            matrix_type,
            typename std::enable_if<
                is_dense_matrix<matrix_type>::value &&
                    std::is_same<typename traits<matrix_type>::backend_type,
                                 blas_backend>::value,
                void>::type>
        {
        public:
            using int_type = blas_backend::int_type;
            using value_type =
                typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
            using backend_type = typename traits<matrix_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using mem_trans = memory::transfer<backend_type, backend_type>;

            using helper = qr_helper<value_type, backend_type>;

        protected:
            tensor<value_type, 1, backend_type> m_tau;
            tensor<value_type, 1, backend_type> m_work;
            tensor<value_type, 2, backend_type> m_mat;
            typename helper::additional_working m_additional;

        public:
            dense_matrix_lq() {}
            dense_matrix_lq(size_type m, size_type n, bool requires_matrix = true)
            {
                CALL_AND_HANDLE(resize(m, n, requires_matrix),
                                "Failed construct singular value decomposition engine with "
                                "preallocated size.  Failed when resizing buffers.");
            }
            template <typename mat_type,
                      typename =
                          typename std::enable_if<is_tensor<mat_type>::value, void>::type,
                      typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
            dense_matrix_lq(const mat_type &mat, bool requires_matrix = true)
            {
                CALL_AND_HANDLE(resize(mat.shape(0), mat.shape(1), requires_matrix),
                                "Failed construct singular value decomposition engine with "
                                "preallocated size.  Failed when resizing buffers.");
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for resizing the internal buffers required for computing a lq //
            // decomposition //
            /////////////////////////////////////////////////////////////////////////////////////////
            void resize(size_type m, size_type n, bool requires_matrix = true)
            {
                try
                {
                    CALL_AND_HANDLE(m_tau.resize(m > n ? n : m),
                                    "Failed when resizing internal matrix.");
                    if (requires_matrix)
                    {
                        CALL_AND_HANDLE(m_mat.resize(m, n),
                                        "Failed when resizing internal matrix.");
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to resize singular value decomposition object.");
                }
            }
            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, void>
            resize(const mat_type &mat, bool requires_matrix = true)
            {
                CALL_AND_RETHROW(resize(mat.shape(0), mat.shape(1), requires_matrix));
            }
            void resize_work_space(size_type worksize)
            {
                if (m_work.size() < worksize)
                {
                    CALL_AND_HANDLE(m_work.resize(worksize),
                                    "Failed to resize workspace of eigensolver object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for clearing the lq decomposition engine
            /////////////////////////////////////////////////////////////////////////////////////////
            void clear()
            {
                try
                {
                    CALL_AND_HANDLE(m_work.clear(), "Failed to clear the work array.");
                    CALL_AND_HANDLE(m_tau.clear(),
                                    "Failed to clear the additional working array.");
                    CALL_AND_HANDLE(m_mat.clear(),
                                    "Failed to clear the additional working array.");
                    CALL_AND_HANDLE(m_additional.clear(),
                                    "Failed to clear the additional working array.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to clear singular value decomposition object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for computing the lq decomposition of a general dense matrix //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename mat_typeb, typename mat_typec>
            valid_decomp_matrix_type_3<matrix_type, mat_type, mat_typeb, mat_typec, void>
            operator()(const mat_type &mat, mat_typeb &L, mat_typec &Q)
            {
                try
                {
                    CALL_AND_HANDLE(resize(mat, true), "Failed to resize internal buffers.");
                    CALL_AND_HANDLE(m_mat = mat,
                                    "Failed to copy input matrix into working buffer.");
                    CALL_AND_RETHROW(compute(m_mat, L, Q));
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate lq.");
                }
            }

            template <typename mat_type, typename mat_typeb, typename mat_typec>
            valid_decomp_matrix_type_3<matrix_type, mat_type, mat_typeb, mat_typec, void>
            operator()(mat_type &mat, mat_typeb &L, mat_typec &Q,
                       bool keep_matrix = true)
            {
                try
                {
                    CALL_AND_HANDLE(resize(mat, !keep_matrix),
                                    "Failed to resize internal buffers.");
                    if (keep_matrix)
                    {
                        CALL_AND_HANDLE(m_mat = mat,
                                        "Failed to copy input matrix into working buffer.");
                        CALL_AND_RETHROW(compute(m_mat, L, Q));
                    }
                    else
                    {
                        CALL_AND_RETHROW(compute(mat, L, Q));
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate lq.");
                }
            }

            template <typename mat_type, typename mat_typeb, typename mat_typec>
            valid_decomp_matrix_type_3<matrix_type, mat_type, mat_typeb, mat_typec, void>
            operator()(const mat_type &mat, mat_typeb &L, mat_typec &Q,
                       linalg::vector<int_type, blas_backend> &jpvt)
            {
                try
                {
                    CALL_AND_HANDLE(resize(mat, true), "Failed to resize internal buffers.");
                    CALL_AND_HANDLE(m_mat = mat,
                                    "Failed to copy input matrix into working buffer.");
                    CALL_AND_RETHROW(compute(m_mat, L, Q, jpvt));
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate lq.");
                }
            }

            template <typename mat_type, typename mat_typeb, typename mat_typec>
            valid_decomp_matrix_type_3<matrix_type, mat_type, mat_typeb, mat_typec, void>
            operator()(mat_type &mat, mat_typeb &L, mat_typec &Q,
                       linalg::vector<int_type, blas_backend> &jpvt,
                       bool keep_matrix = true)
            {
                try
                {
                    CALL_AND_HANDLE(resize(mat, !keep_matrix),
                                    "Failed to resize internal buffers.");
                    if (keep_matrix)
                    {
                        CALL_AND_HANDLE(m_mat = mat,
                                        "Failed to copy input matrix into working buffer.");
                        CALL_AND_RETHROW(compute(m_mat, L, Q, jpvt));
                    }
                    else
                    {
                        CALL_AND_RETHROW(compute(mat, L, Q, jpvt));
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate lq.");
                }
            }

        protected:
            template <typename mat_type, typename mat_typeb>
            void set_L(mat_type &mat, mat_typeb &L)
            {
                try
                {
                    size_type rank = std::min(mat.shape(0), mat.shape(1));
                    // now copy the matrix into the correct location
                    CALL_AND_HANDLE(L.resize(mat.shape(0), rank),
                                    "Failed to resize the R buffer.");
                    L.fill_zeros();
                    for (size_type row = 0; row < mat.shape(0); ++row)
                    {
                        size_type nelems = (row + 1) <= rank ? (row + 1) : rank;
                        std::copy_n(mat.buffer() + row * mat.shape(1), (nelems),
                                    L.buffer() + row * rank);
                        std::fill_n(L.buffer() + row * rank + nelems, rank - (nelems),
                                    value_type(0));
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to set L.");
                }
            }

            template <typename mat_type, typename mat_typeb>
            void set_Q(mat_type &mat, mat_typeb &Q)
            {
                try
                {
                    size_type rank = std::min(mat.shape(0), mat.shape(1));
                    // now copy the matrix into the correct location
                    CALL_AND_HANDLE(Q.resize(rank, mat.shape(1)),
                                    "Failed to resize the unitary buffer.");
                    if (mat.shape(0) == mat.shape(1))
                    {
                        CALL_AND_HANDLE(Q = mat, "Failed to copy matrix into Q array.");
                    }
                    else
                    {
                        for (size_type row = 0; row < rank; ++row)
                        {
                            std::copy_n(mat.buffer() + row * mat.shape(1), mat.shape(1),
                                        Q.buffer() + row * mat.shape(1));
                        }
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to set Q.");
                }
            }

            template <typename mat_type, typename mat_typeb, typename mat_typec>
            void compute(mat_type &mat, mat_typeb &L, mat_typec &Q)
            {
                try
                {
                    size_type rank = std::min(mat.shape(0), mat.shape(1));

                    value_type worksize;
                    int_type lwork = -1;
                    // ensure that the worksize buffer is the correct size.
                    CALL_AND_HANDLE(helper::call_qr(mat.shape(1), mat.shape(0), mat.buffer(),
                                                    mat.shape(1), m_tau.buffer(), &worksize,
                                                    lwork),
                                    "Failed to query worksize for LQ decomposition");
                    size_type iworksize = internal::worksize_as_integer(worksize);
                    CALL_AND_HANDLE(resize_work_space(iworksize),
                                    "Failed to compute singular value decomposition of "
                                    "matrix. Failed to resize the optimal workspace array.");

                    // now actually perform the QR decompositions of A as a column major
                    // matrix (e.g. perform QR of A^T)
                    CALL_AND_HANDLE(helper::call_qr(mat.shape(1), mat.shape(0), mat.buffer(),
                                                    mat.shape(1), m_tau.buffer(),
                                                    m_work.buffer(), m_work.size()),
                                    "Failed to compute L matrix and the elementary "
                                    "reflectors defining the unitary.");

                    CALL_AND_RETHROW(set_L(mat, L));

                    // now resize the workbuffer for constructing the unitary
                    std::cerr << mat.shape(1) << " " << rank << " " << rank << std::endl;
                    CALL_AND_HANDLE(blas_backend::ungqr(mat.shape(1), rank, rank,
                                                        mat.buffer(), mat.shape(1),
                                                        m_tau.buffer(), &worksize, lwork),
                                    "Failed to query worksize for constructing unitary from "
                                    "elementary reflectors.");
                    iworksize = internal::worksize_as_integer(worksize);
                    CALL_AND_HANDLE(resize_work_space(iworksize),
                                    "Failed to compute singular value decomposition of "
                                    "matrix. Failed to resize the optimal workspace array.");
                    CALL_AND_HANDLE(blas_backend::ungqr(
                                        mat.shape(1), rank, rank, mat.buffer(), mat.shape(1),
                                        m_tau.buffer(), m_work.buffer(), m_work.size()),
                                    "Failed to compute unitary from elementary reflectors.");

                    CALL_AND_RETHROW(set_Q(mat, Q));
                }
                catch (const invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating lq.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate lq.");
                }
            }

            template <typename mat_type, typename mat_typeb, typename mat_typec>
            void compute(mat_type &mat, mat_typeb &L, mat_typec &Q,
                         linalg::vector<int_type, blas_backend> &jpvt)
            {
                try
                {
                    size_type rank = std::min(mat.shape(0), mat.shape(1));

                    CALL_AND_HANDLE(m_additional.resize(mat.shape(0), mat.shape(1)),
                                    "Failed to resize additional working space.");
                    value_type worksize;
                    int_type lwork = -1;
                    // ensure that the worksize buffer is the correct size.
                    CALL_AND_HANDLE(helper::call_qr(mat.shape(1), mat.shape(0), mat.buffer(),
                                                    mat.shape(1), jpvt.buffer(),
                                                    m_tau.buffer(), &worksize, lwork,
                                                    m_additional),
                                    "Failed to query worksize for LQ decomposition");
                    size_type iworksize = internal::worksize_as_integer(worksize);
                    CALL_AND_HANDLE(resize_work_space(iworksize),
                                    "Failed to compute singular value decomposition of "
                                    "matrix. Failed to resize the optimal workspace array.");

                    // now actually perform the QR decompositions of A as a column major
                    // matrix (e.g. perform QR of A^T)
                    CALL_AND_HANDLE(helper::call_qr(mat.shape(1), mat.shape(0), mat.buffer(),
                                                    mat.shape(1), jpvt.buffer(),
                                                    m_tau.buffer(), m_work.buffer(),
                                                    m_work.size(), m_additional),
                                    "Failed to compute L matrix and the elementary "
                                    "reflectors defining the unitary.");

                    CALL_AND_RETHROW(set_L(mat, L));

                    // now resize the workbuffer for constructing the unitary
                    std::cerr << mat.shape(1) << " " << rank << " " << rank << std::endl;
                    CALL_AND_HANDLE(blas_backend::ungqr(mat.shape(1), rank, rank,
                                                        mat.buffer(), mat.shape(1),
                                                        m_tau.buffer(), &worksize, lwork),
                                    "Failed to query worksize for constructing unitary from "
                                    "elementary reflectors.");
                    iworksize = internal::worksize_as_integer(worksize);
                    CALL_AND_HANDLE(resize_work_space(iworksize),
                                    "Failed to compute singular value decomposition of "
                                    "matrix. Failed to resize the optimal workspace array.");
                    CALL_AND_HANDLE(blas_backend::ungqr(
                                        mat.shape(1), rank, rank, mat.buffer(), mat.shape(1),
                                        m_tau.buffer(), m_work.buffer(), m_work.size()),
                                    "Failed to compute unitary from elementary reflectors.");

                    CALL_AND_RETHROW(set_Q(mat, Q));
                }
                catch (const invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating lq.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate lq.");
                }
            }
        };

    } // namespace internal

} // namespace linalg

#endif // PYTTN_LINALG_DECOMPOSITIONS_QR_BLAS_HPP_
