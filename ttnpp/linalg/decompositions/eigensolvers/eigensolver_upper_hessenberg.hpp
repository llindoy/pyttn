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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_EIGENSOLVERS_EIGENSOLVER_UPPER_HESSENBERG_HPP_
#define PYTTN_LINALG_DECOMPOSITIONS_EIGENSOLVERS_EIGENSOLVER_UPPER_HESSENBERG_HPP_

#include "eigensolver_base.hpp"

namespace linalg
{
    namespace internal
    {

        template <typename matrix_type, typename enabler = void>
        class upper_hessenberg_eigensolver;

        // implementation of the upper hessenberg eigensolver for real valued matrices
        template <typename matrix_type>
        class upper_hessenberg_eigensolver<matrix_type, typename std::enable_if<!is_complex<typename traits<remove_cvref_t<matrix_type>>::value_type>::value, void>::type>
        {
            static_assert(is_upper_hessenberg_matrix<matrix_type>::value, "Invalid template parameter for upper_hessenberg_eigensolver");

            using int_type = blas_backend::blas_int_type;
            using select_type = blas_backend::select_type;

        public:
            using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
            using backend_type = typename traits<matrix_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using mem_trans = memory::transfer<backend_type, backend_type>;

        protected:
            tensor<value_type, 1, backend_type> m_work;
            tensor<value_type, 2, backend_type> m_mat;
            diagonal_matrix<value_type, backend_type> m_eigs_r;
            diagonal_matrix<value_type, backend_type> m_eigs_i;

        public:
            upper_hessenberg_eigensolver() {}
            upper_hessenberg_eigensolver(size_type n, bool requires_matrix = true) { CALL_AND_HANDLE(resize(n, requires_matrix), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers."); }

            template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
            upper_hessenberg_eigensolver(const mat_type &mat, bool requires_matrix = true) { CALL_AND_HANDLE(resize(mat.shape(0), requires_matrix), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers."); }

            /////////////////////////////////////////////////////////////////////////////////////////
            //     Functions for resizing the internal buffers required for computing a symmetric  //
            //                      tridiagonal matrix eigendecomposition.                         //
            /////////////////////////////////////////////////////////////////////////////////////////
            void resize(size_type n, bool requires_matrix = true)
            {
                try
                {
                    if (requires_matrix)
                    {
                        CALL_AND_HANDLE(m_mat.resize(n, n), "Failed when resizing internal matrix.");
                    }
                    CALL_AND_HANDLE(m_eigs_r.resize(n, n), "Failed to resize the internal real part of the eigenvalues array.");
                    CALL_AND_HANDLE(m_eigs_i.resize(n, n), "Failed to resize the internal imaginary part of the eigenvalues array.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to resize eigensolver object.");
                }
            }
            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type &mat, bool requires_matrix = true) { CALL_AND_RETHROW(resize(mat.shape(0), requires_matrix)); }
            void resize_workspace(size_type worksize)
            {
                if (m_work.size() < worksize)
                {
                    CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of eigensolver object.");
                }
            }

            void clear()
            {
                try
                {
                    CALL_AND_HANDLE(m_mat.clear(), "Failed when clearing internal matrix.");
                    CALL_AND_HANDLE(m_work.clear(), "Failed when clearing the working array.");
                    CALL_AND_HANDLE(m_eigs_r.clear(), "Failed to clear the eigs_r array.");
                    CALL_AND_HANDLE(m_eigs_i.clear(), "Failed to clear the eigs_i array.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to clear eigensolver object.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            //   Functions for computing the eigendecomposition of a upper_hessenberg matrix       //
            //   that do not overwrite the input matrix type.                                      //
            /////////////////////////////////////////////////////////////////////////////////////////
            // functions for computing the eigenvalues of a upper hessenberg matrix
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_vals_type, vals_type> operator()(const mat_type &mat, vals_type &eigs)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, eigs, true));
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                    }
                    CALL_AND_RETHROW(compute(m_mat, eigs));
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating eigenvalues of upper hessenberg matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate eigenvalues of upper hessenberg matrix.");
                }
            }
            template <typename mat_type, typename vals_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_vals_type, vals_type> operator()(mat_type &mat, vals_type &eigs, bool keep_inputs = true)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, eigs, keep_inputs));
                    if (keep_inputs)
                    {
                        if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                        {
                            CALL_AND_HANDLE(m_mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                        }
                        CALL_AND_RETHROW(compute(m_mat, eigs));
                    }
                    else
                    {
                        if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                        {
                            CALL_AND_HANDLE(mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                        }
                        CALL_AND_RETHROW(compute(mat, eigs));
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating eigenvalues of upper hessenberg matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate eigenvalues of upper hessenberg matrix.");
                }
            }

            // functions for computing the eigenvalues and right eigenvectors of a upper hessenberg matrix
            template <typename mat_type, typename vals_type, typename vecs_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_type, vals_type, vecs_type> operator()(const mat_type &mat, vals_type &eigs, vecs_type &vecs)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, eigs, vecs, true));
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                    }
                    CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating eigendecomposition of upper hessenberg matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate eigendecomposition of upper hessenberg matrix.");
                }
            }
            template <typename mat_type, typename vals_type, typename vecs_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_type, vals_type, vecs_type> operator()(mat_type &mat, vals_type &eigs, vecs_type &vecs, bool keep_inputs = true)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, eigs, vecs, keep_inputs));
                    if (keep_inputs)
                    {
                        if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                        {
                            CALL_AND_HANDLE(m_mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                        }
                        CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
                    }
                    else
                    {
                        if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                        {
                            CALL_AND_HANDLE(mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                        }
                        CALL_AND_RETHROW(compute(mat, eigs, vecs));
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating eigendecomposition of upper hessenberg matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate eigendecomposition of upper hessenberg matrix.");
                }
            }

            // functions for computing the eigenvalues and left and right eigenvectors of a upper hessenberg matrix
            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(const mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, eigs, vecs_r, vecs_l, true));
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                    }
                    CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating eigendecomposition of upper hessenberg matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate eigendecomposition of upper hessenberg matrix.");
                }
            }
            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l, bool keep_inputs)
            {
                try
                {
                    CALL_AND_RETHROW(validate(mat, eigs, vecs_r, vecs_l, keep_inputs));
                    if (keep_inputs)
                    {
                        if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                        {
                            CALL_AND_HANDLE(m_mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                        }
                        CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
                    }
                    else
                    {
                        if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                        {
                            CALL_AND_HANDLE(mat = trans(mat), "Failed to transpose the matrix so that it is in column major form.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(mat = mat, "Failed to copy input matrix.");
                        }
                        CALL_AND_RETHROW(compute(mat, eigs, vecs_r, vecs_l));
                    }
                }
                catch (const common::invalid_value &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_NUMERIC("evaluating eigendecomposition of upper hessenberg matrix.");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate eigendecomposition of upper hessenberg matrix.");
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            //  Functions for determining the optimal workspace for eigendecomposition             //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, size_type> query_work_space(mat_type &mat)
            {
                char JOB = 'E';
                char COMPZ = 'N';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                value_type Z;
                int_type LDZ = 1;
                value_type worksize;
                int_type LWORK = -1;
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, m_eigs_r.buffer(), m_eigs_i.buffer(), &Z, LDZ, &worksize, LWORK), "Failed to query the optimal worksize for hseqr call.");
                size_type iworksize;
                CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
                return (iworksize > 4 * mat.shape(0) ? iworksize : 4 * mat.shape(0));
            }

            template <typename mat_type, typename vecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_complex_vecs_type, vecs_type> query_work_space(mat_type &mat, vecs_type &vecs)
            {
                value_type *vecsb = reinterpret_cast<value_type *>(vecs.buffer());
                char JOB = 'S';
                char COMPZ = 'I';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                int_type LDZ = vecs.shape(1);
                value_type worksize;
                int_type LWORK = -1;
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, m_eigs_r.buffer(), m_eigs_i.buffer(), vecsb, LDZ, &worksize, LWORK), "Failed to query the optimal worksize for hseqr call.");
                size_type iworksize;
                CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
                return (iworksize > 4 * mat.shape(0) ? iworksize : 4 * mat.shape(0));
            }

        protected:
            /////////////////////////////////////////////////////////////////////////////////////////
            //  Helper Functions for computing the eigendecomposition of a upper hessenberg matrix //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            void compute(mat_type &mat, vals_type &eigs)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space(mat), "Failed to compute eigenvalues of upper_hessenberg matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of upper hessenberg matrix. Failed to resize the optimal workspace array.");

                // now we make the hseqr call to evaluate the eigenvalues
                char JOB = 'E';
                char COMPZ = 'N';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                value_type Z;
                int_type LDZ = 1;
                int_type LWORK = m_work.size();
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, m_eigs_r.buffer(), m_eigs_i.buffer(), &Z, LDZ, m_work.buffer(), LWORK), "Failed to compute eigenvalues of upper_hessenberg matrix.  Failed when calling hseqr.");
                CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_eigs_r.buffer(), 1, m_eigs_i.buffer(), 1, eigs.buffer(), eigs.incx()), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to interleave real and imaginary part arrays to form complex array.");
            }

            template <typename mat_type, typename vals_type, typename vecs_type>
            void compute(mat_type &mat, vals_type &eigs, vecs_type &vecs)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space(mat, vecs), "Failed to compute eigenvalues of upper_hessenberg matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of upper hessenberg matrix. Failed to resize the optimal workspace array.");

                value_type *rvecs = reinterpret_cast<value_type *>(vecs.buffer());
                char JOB = 'S';
                char COMPZ = 'I';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                int_type LDZ = vecs.shape(1);
                int_type LWORK = m_work.size();
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, m_eigs_r.buffer(), m_eigs_i.buffer(), rvecs, LDZ, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed when computing the eigenvalues using hseqr.");
                CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_eigs_r.buffer(), 1, m_eigs_i.buffer(), 1, eigs.buffer(), eigs.incx()), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to interleave real and imaginary part arrays to form complex array.");

                char SIDE = 'R';
                char HOWMNY = 'B';
                select_type select;
                int_type LDT = mat.shape(1);
                value_type VL;
                int_type LDVL = 1;
                int_type LDVR = vecs.shape(1);
                int_type MM = 2 * vecs.shape(1);
                CALL_AND_HANDLE(backend_type::trevc(SIDE, HOWMNY, &select, N, mat.buffer(), LDT, &VL, LDVL, rvecs, LDVR, MM, m_work.buffer()), "Failed to compute eigenvectors of upper_hessenberg matrix.  Failed when computing the eigenvectors using trevc.");

                CALL_AND_HANDLE(mem_trans::copy(rvecs, N * N, mat.buffer()), "Failed to copy the packed eigenvector buffer to the mat buffer.");
                CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_eigs_i.buffer(), mat.buffer(), vecs.buffer()), "Failed to unpack eigenvectors buffer to complex format.");
                CALL_AND_HANDLE(vecs = trans(vecs), "Failed to compute eigenvectors of upper hessenberg matrix.  Failed to transpose eigenvalues.");
            }

            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            void compute(mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space(m_mat, vecs_r), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of upper hessenberg matrix. Failed to resize the optimal workspace array.");

                value_type *rvecsr = reinterpret_cast<value_type *>(vecs_r.buffer());
                value_type *rvecsl = reinterpret_cast<value_type *>(vecs_l.buffer());
                char JOB = 'S';
                char COMPZ = 'I';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                int_type LDZ = vecs_r.shape(1);
                int_type LWORK = m_work.size();
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, m_eigs_r.buffer(), m_eigs_i.buffer(), rvecsr, LDZ, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed when computing the eigenvalues using hseqr.");
                CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_eigs_r.buffer(), 1, m_eigs_i.buffer(), 1, eigs.buffer(), eigs.incx()), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to interleave real and imaginary part arrays to form complex array.");

                CALL_AND_HANDLE(vecs_l = vecs_r, "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to copy the unitary matrix returned by hseqr into the left eigenvector matrix.");

                char SIDE = 'B';
                char HOWMNY = 'B';
                select_type select;
                int_type LDT = mat.shape(1);
                int_type LDVL = vecs_l.shape(1);
                int_type LDVR = vecs_r.shape(1);
                int_type MM = vecs_r.shape(1);
                CALL_AND_HANDLE(backend_type::trevc(SIDE, HOWMNY, &select, N, mat.buffer(), LDT, rvecsl, LDVL, rvecsr, LDVR, MM, m_work.buffer()), "Failed to compute eigenvectors of upper_hessenberg matrix.  Failed when computing the eigenvectors using trevc.");

                CALL_AND_HANDLE(mem_trans::copy(rvecsr, N * N, mat.buffer()), "Failed to copy the packed eigenvector buffer to the mat buffer.");
                CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_eigs_i.buffer(), mat.buffer(), vecs_r.buffer()), "Failed to unpack eigenvectors buffer to complex format.");

                CALL_AND_HANDLE(mem_trans::copy(rvecsl, N * N, mat.buffer()), "Failed to copy the packed eigenvector buffer to the mat buffer.");
                CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_eigs_i.buffer(), mat.buffer(), vecs_l.buffer()), "Failed to unpack eigenvectors buffer to complex format.");
                // take the inner product of the left and right eigenvectors to allow for a sensible normalisation
                CALL_AND_HANDLE(for (size_type i = 0; i < vecs_r.shape(0); ++i) {complex<value_type> scaling = static_cast<value_type>(1.0)/dot_product(conj(vecs_l[i]), vecs_r[i]); vecs_r[i] *= scaling; }, "Failed to compute eigenvectors of upper hessenberg matrix.  Failed to rescale right eigenvectors.");
                CALL_AND_HANDLE(vecs_r = trans(vecs_r), "Failed to compute eigenvectors of upper hessenberg matrix.  Failed to transpose right eigenvectors.");
                CALL_AND_HANDLE(vecs_l = trans(vecs_l), "Failed to compute eigenvectors of upper hessenberg matrix.  Failed to transpose left eigenvectors.");
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
            //  engine for a upper hessenberg matrix                                               //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            void validate(mat_type &&mat, vals_type &eigs, bool use_matrix)
            {
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvalues array.");
                CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename vecs_type>
            void validate(mat_type &mat, vals_type &eigs, vecs_type &vecs, bool use_matrix)
            {
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvalues array.");
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
                CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            void validate(mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l, bool use_matrix)
            {
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvalues array.");
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_r), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to ensure the correct shape of the right eigenvector matrix.");
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_l), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to ensure the correct shape of the left eigenvector matrix.");
                CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed resize buffers.");
            }
        };

        // implementation of the upper hessenberg eigensolver for complex valued matrices
        template <typename matrix_type>
        class upper_hessenberg_eigensolver<matrix_type, typename std::enable_if<is_complex<typename traits<remove_cvref_t<matrix_type>>::value_type>::value, void>::type>
        {
            static_assert(is_upper_hessenberg_matrix<matrix_type>::value, "Invalid template parameter for upper_hessenberg_eigensolver");
            using int_type = blas_backend::blas_int_type;
            using select_type = blas_backend::select_type;

        public:
            using value_type = typename traits<matrix_type>::value_type;
            using backend_type = typename traits<matrix_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using mem_trans = memory::transfer<backend_type, backend_type>;

            static_assert(std::is_same<backend_type, linalg::blas_backend>::value, "upperhessenberg eigensolver only valid for blas backend.");

        protected:
            tensor<typename get_real_type<value_type>::type, 1, backend_type> m_rwork;
            tensor<value_type, 1, backend_type> m_work;
            tensor<value_type, 2, backend_type> m_mat;
            diagonal_matrix<value_type, backend_type> m_scaling;

        public:
            upper_hessenberg_eigensolver() {}
            upper_hessenberg_eigensolver(size_type n, bool requires_matrix = true, bool requires_scaling = false) { CALL_AND_HANDLE(resize(n, requires_matrix, requires_scaling), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers."); }
            template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
            upper_hessenberg_eigensolver(const mat_type &mat, bool requires_matrix = true, bool requires_scaling = false) { CALL_AND_HANDLE(resize(mat.shape(0), requires_matrix, requires_scaling), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers."); }

            /////////////////////////////////////////////////////////////////////////////////////////
            //     Functions for resizing the internal buffers required for computing a symmetric  //
            //                      tridiagonal matrix eigendecomposition.                         //
            /////////////////////////////////////////////////////////////////////////////////////////
            void resize(size_type n, bool requires_matrix = true, bool requires_scaling = false)
            {
                if (requires_matrix)
                {
                    CALL_AND_HANDLE(m_mat.resize(n, n), "Failed to resize eigensolver object.  Failed when resizing internal matrix.");
                }
                if (requires_scaling)
                {
                    CALL_AND_HANDLE(m_scaling.resize(n, n), "Failed to resize eigensolver object.  Failed when resizing the internal scaling matrix required to give sensibly normalised eigenvectors.");
                }
                CALL_AND_HANDLE(m_rwork.resize(n), "Failed to resize eigensolver object.  Failed when resizing the rwork working space.");
            }
            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type &mat, bool requires_matrix = true, bool requires_scaling = false) { CALL_AND_RETHROW(resize(mat.shape(0), requires_matrix, requires_scaling)); }
            void resize_workspace(size_type worksize)
            {
                if (m_work.size() < worksize)
                {
                    CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of eigensolver object.");
                }
            }

            void clear()
            {
                CALL_AND_HANDLE(m_mat.clear(), "Failed to clear eigensolver object.  Failed when clearing internal matrix.");
                CALL_AND_HANDLE(m_work.clear(), "Failed to clear eigensolver object.  Failed when clearing the working array.");
                CALL_AND_HANDLE(m_rwork.clear(), "Failed to clear eigensolver object.  Failed when clearing the real working array.");
                CALL_AND_HANDLE(m_scaling.clear(), "Failed to clear eigensolver object.  Failed to clear the scaling array.");
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            //   Functions for computing the eigendecomposition of a upper_hessenberg matrix       //
            //   that do not overwrite the input matrix type.                                      //
            /////////////////////////////////////////////////////////////////////////////////////////
            // functions for computing the eigenvalues of a upper hessenberg matrix
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_vals_type, vals_type> operator()(const mat_type &mat, vals_type &eigs)
            {
                CALL_AND_RETHROW(validate(mat, eigs, true));
                if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                {
                    CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                }
                else
                {
                    CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of upper_hessenberg matrix.  Failed to copy input matrix.");
                }
                CALL_AND_RETHROW(compute(m_mat, eigs));
            }
            template <typename mat_type, typename vals_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_vals_type, vals_type> operator()(mat_type &mat, vals_type &eigs, bool keep_inputs = true)
            {
                CALL_AND_RETHROW(validate(mat, eigs, keep_inputs));
                if (keep_inputs)
                {
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of upper_hessenberg matrix.  Failed to copy input matrix.");
                    }
                    CALL_AND_RETHROW(compute(m_mat, eigs));
                }
                else
                {
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(mat = trans(mat), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                    }
                    CALL_AND_RETHROW(compute(mat, eigs));
                }
            }

            // functions for computing the eigenvalues and right eigenvectors of a upper hessenberg matrix
            template <typename mat_type, typename vals_type, typename vecs_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_type, vals_type, vecs_type> operator()(const mat_type &mat, vals_type &eigs, vecs_type &vecs)
            {
                CALL_AND_RETHROW(validate(mat, eigs, vecs, true));
                if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                {
                    CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                }
                else
                {
                    CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to copy input matrix.");
                }
                CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
            }
            template <typename mat_type, typename vals_type, typename vecs_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_type, vals_type, vecs_type> operator()(mat_type &mat, vals_type &eigs, vecs_type &vecs, bool keep_inputs = true)
            {
                CALL_AND_RETHROW(validate(mat, eigs, vecs, keep_inputs));
                if (keep_inputs)
                {
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to copy input matrix.");
                    }
                    CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
                }
                else
                {
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(mat = trans(mat), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                    }
                    CALL_AND_RETHROW(compute(mat, eigs, vecs));
                }
            }

            // functions for computing the eigenvalues, and left and right eigenvectors of a upper hessenberg matrix
            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(const mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l)
            {
                CALL_AND_RETHROW(validate(mat, eigs, vecs_r, vecs_l, true));
                if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                {
                    CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                }
                else
                {
                    CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to copy input matrix.");
                }
                CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
            }
            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            valid_mutable_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l, bool keep_inputs = true)
            {
                CALL_AND_RETHROW(validate(mat, eigs, vecs_r, vecs_l, keep_inputs));
                if (keep_inputs)
                {
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to copy input matrix.");
                    }
                    CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
                }
                else
                {
                    if (mat.order() == MATRIX_ORDERING::ROW_MAJOR)
                    {
                        CALL_AND_HANDLE(mat = trans(mat), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose the matrix so that it is in column major form.");
                    }
                    CALL_AND_RETHROW(compute(mat, eigs, vecs_r, vecs_l));
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            //  Functions for determining the optimal workspace for eigendecomposition             //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_type, vals_type> query_work_space(mat_type &mat, vals_type &eigs)
            {
                char JOB = 'E';
                char COMPZ = 'N';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                value_type Z;
                int_type LDZ = 1;
                value_type worksize;
                int_type LWORK = -1;
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, eigs.buffer(), &Z, LDZ, &worksize, LWORK), "Failed to query the optimal worksize for hseqr call.");
                size_type iworksize;
                CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
                return (iworksize > 4 * mat.shape(0) ? iworksize : 4 * mat.shape(0));
            }

            template <typename mat_type, typename vals_type, typename vecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_vecs_type, vals_type, vecs_type> query_work_space(mat_type &mat, vals_type &eigs, vecs_type &vecs)
            {
                char JOB = 'S';
                char COMPZ = 'I';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                int_type LDZ = vecs.shape(1);
                value_type worksize;
                int_type LWORK = -1;
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, eigs.buffer(), vecs.buffer(), LDZ, &worksize, LWORK), "Failed to query the optimal worksize for hseqr call.");
                size_type iworksize;
                CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
                return (iworksize > 4 * mat.shape(0) ? iworksize : 4 * mat.shape(0));
            }

        protected:
            /////////////////////////////////////////////////////////////////////////////////////////
            //  Helper Functions for computing the eigendecomposition of a upper hessenberg matrix //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            void compute(mat_type &mat, vals_type &eigs)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space(mat, eigs), "Failed to compute eigenvalues of upper_hessenberg matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of upper hessenberg matrix. Failed to resize the optimal workspace array.");
                char JOB = 'E';
                char COMPZ = 'N';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                value_type Z;
                int_type LDZ = 1;
                int_type LWORK = m_work.size();
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, eigs.buffer(), &Z, LDZ, m_work.buffer(), LWORK), "Failed to compute eigenvalues of upper_hessenberg matrix.  Failed when calling hseqr.");
            }

            template <typename mat_type, typename vals_type, typename vecs_type>
            void compute(mat_type &mat, vals_type &eigs, vecs_type &vecs)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space(mat, eigs, vecs), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of upper hessenberg matrix. Failed to resize the optimal workspace array.");

                char JOB = 'S';
                char COMPZ = 'I';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                int_type LDZ = vecs.shape(1);
                int_type LWORK = m_work.size();
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, eigs.buffer(), vecs.buffer(), LDZ, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed when computing the eigenvalues using hseqr.");

                char SIDE = 'R';
                char HOWMNY = 'B';
                select_type select;
                int_type LDT = mat.shape(1);
                value_type VL;
                int_type LDVL = 1;
                int_type LDVR = vecs.shape(1);
                int_type MM = vecs.shape(1);
                CALL_AND_HANDLE(backend_type::trevc(SIDE, HOWMNY, &select, N, mat.buffer(), LDT, &VL, LDVL, vecs.buffer(), LDVR, MM, m_work.buffer(), m_rwork.buffer()), "Failed to compute eigenvectors of upper_hessenberg matrix.  Failed when computing the eigenvectors using trevc.");

                // now vecs is in column major order so we convert it to row major order
                CALL_AND_HANDLE(vecs = trans(vecs), "Failed to compute eigenvectors of upper hessenberg matrix.  Failed to transpose eigenvalues.");
            }

            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            void compute(mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l)
            {
                size_type worksize;
                CALL_AND_HANDLE(worksize = query_work_space(mat, eigs, vecs_r), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to determine optimal work space.");
                CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of upper hessenberg matrix. Failed to resize the optimal workspace array.");

                char JOB = 'S';
                char COMPZ = 'I';
                int_type N = mat.shape(0);
                int_type ILO = 1;
                int_type IHI = N;
                int_type LDH = mat.shape(1);
                int_type LDZ = vecs_r.shape(1);
                int_type LWORK = m_work.size();
                CALL_AND_HANDLE(backend_type::hseqr(JOB, COMPZ, N, ILO, IHI, mat.buffer(), LDH, eigs.buffer(), vecs_r.buffer(), LDZ, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed when computing the eigenvalues using hseqr.");

                CALL_AND_HANDLE(vecs_l = vecs_r, "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed to copy the unitary matrix returned by hseqr into the left eigenvector matrix.");

                char SIDE = 'B';
                char HOWMNY = 'B';
                select_type select;
                int_type LDT = mat.shape(1);
                int_type LDVL = vecs_l.shape(1);
                int_type LDVR = vecs_r.shape(1);
                int_type MM = vecs_r.shape(1);
                CALL_AND_HANDLE(backend_type::trevc(SIDE, HOWMNY, &select, N, mat.buffer(), LDT, vecs_l.buffer(), LDVL, vecs_r.buffer(), LDVR, MM, m_work.buffer(), m_rwork.buffer()), "Failed to compute eigendecomposition of upper_hessenberg matrix.  Failed when computing the eigenvectors using trevc.");

                // take the inner product of the left and right eigenvectors to allow for a sensible normalisation
                CALL_AND_HANDLE(using std::sqrt; for (size_type i = 0; i < vecs_r.shape(0); ++i) { m_scaling[i] = static_cast<value_type>(1.0) / dot_product(conj(vecs_l[i]), vecs_r[i]); }, "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to compute the rescaling factor.");
                CALL_AND_HANDLE(mat = m_scaling * vecs_r, "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to rescale right eigendecomposition.");
                CALL_AND_HANDLE(vecs_r = trans(mat), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose right eigenvectors.");
                CALL_AND_HANDLE(vecs_l = trans(vecs_l), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to transpose left eigenvectors.");
            }

            /////////////////////////////////////////////////////////////////////////////////////////
            //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
            //  engine for a upper hessenberg matrix                                               //
            /////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            void validate(const mat_type &mat, vals_type &eigs, bool use_matrix)
            {
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvalues array.");
                CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename vecs_type>
            void validate(const mat_type &mat, vals_type &eigs, vecs_type &vecs, bool use_matrix)
            {
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvalues array.");
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
                CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed resize buffers.");
            }

            template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
            void validate(const mat_type &mat, vals_type &eigs, rvecs_type &vecs_r, lvecs_type &vecs_l, bool use_matrix)
            {
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of upper hessenberg matrix.  Failed to ensure the correct shape of the eigenvalues array.");
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_r), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to ensure the correct shape of the right eigenvector matrix.");
                CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_l), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed to ensure the correct shape of the left eigenvector matrix.");
                CALL_AND_HANDLE(resize(mat.shape(0), use_matrix, true), "Failed to compute eigendecomposition of upper hessenberg matrix.  Failed resize buffers.");
            }
        };
    } // namespace internal

    // eigensolver object for upper hessenberg matrix type that is mutable.  This function provides
    // additional evaluating routines that optionally overwrite the input matrix.
    template <typename matrix_type>
    class eigensolver<matrix_type,
                      typename std::enable_if<
                          is_dense_matrix<remove_cvref_t<matrix_type>>::value && is_upper_hessenberg_matrix<remove_cvref_t<matrix_type>>::value &&
                              std::is_same<typename traits<remove_cvref_t<matrix_type>>::backend_type, blas_backend>::value,
                          void>::type> : public internal::upper_hessenberg_eigensolver<remove_cvref_t<matrix_type>>
    {
    public:
        using base_type = internal::upper_hessenberg_eigensolver<remove_cvref_t<matrix_type>>;
        using backend_type = typename traits<remove_cvref_t<matrix_type>>::backend_type;
        using size_type = typename base_type::size_type;
        using value_type = typename base_type::value_type;
        template <typename... Args>
        eigensolver(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (...)
        {
            throw;
        }
    };
} // namespace linalg

#endif // PYTTN_LINALG_DECOMPOSITIONS_EIGENSOLVERS_EIGENSOLVER_UPPER_HESSENBERG_HPP_//
