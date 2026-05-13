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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_BASE_HPP_
#define PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_BASE_HPP_

#include "../decompositions_common.hpp"

namespace linalg
{

    enum svd_result_ordering
    {
        usv,
        vsu
    };

    namespace internal
    {
        struct svd_result_validation
        {
            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_diagonal_matrix_type<vals_type>::value && traits<vals_type>::is_resizable, void>::type singular_values(const matrix_type &mat, vals_type &eigs)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                if ((eigs.shape(0) != minmn) || (mat.shape(1) != minmn))
                {
                    // print out an info statement indicating that the eigensolver is resizing the eigenvalues array.
                    CALL_AND_HANDLE(eigs.resize(minmn, minmn), "Failed to reshape the singular values vector.");
                }
            }
            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_diagonal_matrix_type<vals_type>::value && !traits<vals_type>::is_resizable, void>::type singular_values(const matrix_type &mat, vals_type &eigs)
            {
                ASSERT(((eigs.shape(0) == mat.shape(0)) && (mat.shape(1) == eigs.shape(1))), "The input singular values array does not have the correct shape.");
            }

            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_dense_tensor<vals_type>::value && traits<vals_type>::rank == 1 && traits<vals_type>::is_resizable, void>::type singular_values(const matrix_type &mat, vals_type &eigs)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                if (minmn != eigs.shape(0))
                {
                    // print out an info statement indicating that the eigensolver is resizing the singular values array.
                    CALL_AND_HANDLE(eigs.resize(minmn), "Failed to reshape the singular values vector.");
                }
            }
            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_dense_tensor<vals_type>::value && traits<vals_type>::rank == 1 && !traits<vals_type>::is_resizable, void>::type singular_values(const matrix_type &mat, vals_type &eigs)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                ASSERT(minmn == eigs.shape(0), "The input singular values array does not have the correct shape.");
            }

            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<traits<vecs_type>::is_mutable, void>::type left_singular_vectors(const matrix_type &mat, vecs_type &U)
            {
                if (U.shape(0) != mat.shape(0) || U.shape(1) != mat.shape(0))
                {
                    // print out an info statement indicating that the eigensolver is resizing the eigenvector matrix.
                    CALL_AND_HANDLE(U.resize(mat.shape(0), mat.shape(0)), "Failed to reshape the left singular vectors matrix.");
                }
            }
            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<!traits<vecs_type>::is_mutable, void>::type left_singular_vectors(const matrix_type &mat, vecs_type &U)
            {
                ASSERT(U.shape(0) == mat.shape(0) && U.shape(1) == mat.shape(0), "The input left singular vector matrix does not have the correct shape.");
            }

            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<traits<vecs_type>::is_mutable, void>::type right_singular_vectors(const matrix_type &mat, vecs_type &VT)
            {
                if (VT.shape(0) != mat.shape(1) || VT.shape(1) != mat.shape(1))
                {
                    // print out an info statement indicating that the eigensolver is resizing the eigenvector matrix.
                    CALL_AND_HANDLE(VT.resize(mat.shape(1), mat.shape(1)), "Failed to reshape the right singular vectors matrix.");
                }
            }
            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<!traits<vecs_type>::is_mutable, void>::type right_singular_vectors(const matrix_type &mat, vecs_type &VT)
            {
                ASSERT(VT.shape(0) == mat.shape(1) && VT.shape(1) == mat.shape(1), "The input left singular vector matrix does not have the correct shape.");
            }

            // validation functions for computing the minimal number of singular values and vectors required to fill the tensor
            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_diagonal_matrix_type<vals_type>::value && traits<vals_type>::is_resizable, void>::type minimal_singular_values(const matrix_type &mat, vals_type &eigs)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                if (eigs.shape(0) != minmn && eigs.shape(1) != minmn)
                {
                    CALL_AND_HANDLE(eigs.resize(minmn, minmn), "Failed to reshape the singular values vector.");
                }
            }
            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_diagonal_matrix_type<vals_type>::value && !traits<vals_type>::is_resizable, void>::type minimal_singular_values(const matrix_type &mat, vals_type &eigs)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                ASSERT(((eigs.shape(0) == minmn) && (mat.shape(1) == minmn)), "The input singular values array does not have the correct shape.");
            }

            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_dense_tensor<vals_type>::value && traits<vals_type>::rank == 1 && traits<vals_type>::is_resizable, void>::type minimal_singular_values(const matrix_type &mat, vals_type &eigs)
            {
                CALL_AND_RETHROW(singular_values(mat, eigs));
            }
            template <typename matrix_type, typename vals_type>
            static inline typename std::enable_if<is_dense_tensor<vals_type>::value && traits<vals_type>::rank == 1 && !traits<vals_type>::is_resizable, void>::type minimal_singular_values(const matrix_type &mat, vals_type &eigs)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                ASSERT(minmn == eigs.shape(0), "The input singular values array does not have the correct shape.");
            }

            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<traits<vecs_type>::is_mutable, void>::type minimal_left_singular_vectors(const matrix_type &mat, vecs_type &U)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                if (U.shape(0) != mat.shape(0) || U.shape(1) != minmn)
                {
                    // print out an info statement indicating that the eigensolver is resizing the eigenvector matrix.
                    CALL_AND_HANDLE(U.resize(mat.shape(0), minmn), "Failed to reshape the left singular vectors matrix.");
                }
            }

            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<!traits<vecs_type>::is_mutable, void>::type minimal_left_singular_vectors(const matrix_type &mat, vecs_type &U)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                ASSERT(U.shape(0) == mat.shape(0) && U.shape(1) == minmn, "The input left singular vector matrix does not have the correct shape.");
            }

            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<traits<vecs_type>::is_mutable, void>::type minimal_right_singular_vectors(const matrix_type &mat, vecs_type &VT)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                if (VT.shape(0) != minmn || VT.shape(1) != mat.shape(1))
                {
                    // print out an info statement indicating that the eigensolver is resizing the eigenvector matrix.
                    CALL_AND_HANDLE(VT.resize(minmn, mat.shape(1)), "Failed to reshape the right singular vectors matrix.");
                }
            }
            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<!traits<vecs_type>::is_mutable, void>::type minimal_right_singular_vectors(const matrix_type &mat, vecs_type &VT)
            {
                auto minmn = (mat.shape(0) < mat.shape(1) ? mat.shape(0) : mat.shape(1));
                ASSERT(VT.shape(0) == minmn && VT.shape(1) == mat.shape(1), "The input left singular vector matrix does not have the correct shape.");
            }

            template <typename matrix_type, typename vecs_type>
            static inline typename std::enable_if<traits<vecs_type>::is_mutable, void>::type inplace_singular_vectors(const matrix_type &mat, vecs_type &R)
            {
                auto M = mat.shape(1);
                auto N = mat.shape(0);
                if (M < N)
                {
                    if (R.shape(0) != M || R.shape(1) != M)
                    {
                        CALL_AND_HANDLE(R.resize(M, M), "Failed to resizes the right singular vector matrix for the inplace transform.");
                    }
                }
                else
                {
                    if (R.shape(0) != N || R.shape(1) != N)
                    {
                        CALL_AND_HANDLE(R.resize(N, N), "Failed to resizes the left singular vector matrix for the inplace transform.");
                    }
                }
            }

        }; // struct svd_result_validation

        template <typename T, typename backend, bool use_divide_and_conquer = true>
        struct singular_value_decomposition_helper;
        
        template <typename matrix_type, bool use_divide_and_conquer = true, typename enabler = void>
        class dense_matrix_singular_value_decomposition
        {
            using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
            using backend_type = typename traits<matrix_type>::backend_type;
            using int_type = typename traits<backend_type>::int_type;

            using size_type = typename traits<backend_type>::size_type;
            using mem_trans = memory::transfer<backend_type, backend_type>;
        public:
            dense_matrix_singular_value_decomposition();
            dense_matrix_singular_value_decomposition(size_type m, size_type n, bool requires_matrix = true) ;
            template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
            dense_matrix_singular_value_decomposition(const mat_type &mat, bool requires_matrix = true);

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for resizing the internal buffers required for computing a singular values//
            // decomposition                                                                       //
            /////////////////////////////////////////////////////////////////////////////////////////
            void resize(size_type m, size_type n, bool /*requires_matrix*/ = true); 
            
            template <typename mat_type>
            valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type &mat, bool requires_matrix = true);
            void resize_work_space(size_type worksize);

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for clearing the single value decomposition engine
            /////////////////////////////////////////////////////////////////////////////////////////
            void clear();

            /////////////////////////////////////////////////////////////////////////////////////////
            // Functions for computing the singular values decomposition of a general dense matrix //
            /////////////////////////////////////////////////////////////////////////////////////////

            // Functions for just computing the singular values
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(const mat_type &mat, vals_type &S);

            // Functions for computing the singular values and left and right singular vectors
            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> operator()(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true);

            //////////////////////////////////////////////////////////////////////////////////////////
            //  Functions for determining the optimal workspace for singular valuesdecomposition    //
            //////////////////////////////////////////////////////////////////////////////////////////
            template <typename mat_type, typename vals_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_type, vals_type> query_work_space(const mat_type &mat, vals_type &S);

            template <typename mat_type, typename vals_type, typename uvecs_type, typename vvecs_type>
            valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_vecs_rl_type, vals_type, uvecs_type, vvecs_type> query_work_space(const mat_type &mat, vals_type &S, uvecs_type &U, vvecs_type &Vh, bool compute_full = true);
        };

    } // namespace internal
} // namespace linalg

#endif // PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_BASE_HPP_
