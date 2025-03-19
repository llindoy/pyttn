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

#ifndef PYTTN_LINALG_ALGEBRA_OVERLOADS_MATRIX_MATRIX_PRODUCT_HPP_
#define PYTTN_LINALG_ALGEBRA_OVERLOADS_MATRIX_MATRIX_PRODUCT_HPP_

// TODO combine with the tensor contractions overloads.

#include "hadamard.hpp"

namespace linalg
{
    // matrix - matrix product
    template <typename T1, typename T2, typename... Args>
    gemm_return_type<T1, T2> matmul(const T1 &l, const T2 &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l, r, std::forward<Args>(args)...));
    }

    // trans matrix - matrix product
    template <typename T1, typename T2, bool conjugate, typename... Args>
    gemm_return_type<T1, T2> matmul(const trans_type<T1, conjugate> &l, const T2 &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.coeff()), l.matrix(), r, std::forward<Args>(args)..., true, false, conjugate));
    }

    // matrix - trans matrix product
    template <typename T1, typename T2, bool conjugate, typename... Args>
    gemm_return_type<T1, T2> matmul(const T1 &l, const trans_type<T2, conjugate> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(r.coeff()), l, r.matrix(), std::forward<Args>(args)..., false, true, false, conjugate));
    }

    // trans matrix - trans matrix product
    template <typename T1, typename T2, bool conjugate1, bool conjugate2, typename... Args>
    gemm_return_type<T1, T2> matmul(const trans_type<T1, conjugate1> &l, const trans_type<T2, conjugate2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.coeff() * r.coeff()), l.matrix(), r.matrix(), std::forward<Args>(args)..., true, true, conjugate1, conjugate2));
    }

    // matrix - scalar matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    scal_gemm_return_type<T1, T2, T3> matmul(const T3 &l, const scal_type<T1, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T2, T3>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(r.left()), l, r.right(), std::forward<Args>(args)...));
    }

    // scalar matrix - matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    scal_gemm_return_type<T1, T2, T3> matmul(const scal_type<T1, T2> &l, const T3 &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T2, T3>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.left()), l.right(), r, std::forward<Args>(args)...));
    }

    // scalar matrix - scalar matrix product
    template <typename T1, typename T2, typename T3, typename T4, typename... Args>
    scal_scal_gemm_return_type<T1, T2, T3, T4> matmul(const scal_type<T1, T2> &l, const scal_type<T3, T4> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T2, T4>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T1>(l.left()) * static_cast<T3>(r.left())), l.right(), r.right(), std::forward<Args>(args)...));
    }

    // matrix conj - matrix product
    template <typename T1, typename T2, typename... Args>
    gemm_return_type<T1, T2> matmul(const T1 &l, const conj_type<T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l, r.obj(), std::forward<Args>(args)..., false, false, false, true));
    }

    // conj matrix - matrix product
    template <typename T1, typename T2, typename... Args>
    gemm_return_type<T1, T2> matmul(const conj_type<T1> &l, const T2 &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l.obj(), r, std::forward<Args>(args)..., false, false, true, false));
    }

    // conj matrix - conj matrix product
    template <typename T1, typename T2, typename... Args>
    gemm_return_type<T1, T2> matmul(const conj_type<T1> &l, const conj_type<T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l.obj(), r.obj(), std::forward<Args>(args)..., false, false, true, true));
    }

    // conj matrix - scalar matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    gemm_return_type<T1, T2> matmul(const conj_type<T1> &l, const scal_type<T3, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(r.left()), l.obj(), r.right(), std::forward<Args>(args)..., false, false, true, false));
    }

    // scalar matrix - conj matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    gemm_return_type<T1, T2> matmul(const scal_type<T3, T1> &l, const conj_type<T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.left()), l.right(), r.obj(), std::forward<Args>(args)..., false, false, false, true));
    }

    // trans matrix - scalar matrix product
    template <typename T1, typename T2, typename T3, bool conjugate, typename... Args>
    gemm_return_type<T1, T2> matmul(const trans_type<T1, conjugate> &l, const scal_type<T3, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T3>(r.left()) * l.coeff()), l.matrix(), r.right(), std::forward<Args>(args)..., true, false, conjugate, false));
    }

    // scalar matrix - trans matrix product
    template <typename T1, typename T2, typename T3, bool conjugate, typename... Args>
    gemm_return_type<T1, T2> matmul(const scal_type<T3, T1> &l, const trans_type<T2, conjugate> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T3>(l.left()) * r.coeff()), l.right(), r.matrix(), std::forward<Args>(args)..., false, true, false, conjugate));
    }

    // trans matrix - conj matrix product
    template <typename T1, typename T2, bool conjugate, typename... Args>
    gemm_return_type<T1, T2> matmul(const trans_type<T1, conjugate> &l, const conj_type<T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.coeff()), l.matrix(), r.obj(), std::forward<Args>(args)..., true, false, conjugate, true));
    }

    // scalar matrix - trans matrix product
    template <typename T1, typename T2, bool conjugate, typename... Args>
    gemm_return_type<T1, T2> matmul(const conj_type<T1> &l, const trans_type<T2, conjugate> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(r.coeff()), l.obj(), r.matrix(), std::forward<Args>(args)..., false, true, true, conjugate));
    }

    // matrix - scalar*conj matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    scal_gemm_return_type<T3, T1, T2> matmul(const T1 &l, const scalconj_type<T3, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(r.left()), l, r.right().obj(), std::forward<Args>(args)..., false, false, false, true));
    }

    // scalar*conj matrix - matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    scal_gemm_return_type<T3, T1, T2> matmul(const scalconj_type<T3, T1> &l, const T2 &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.left()), l.right().obj(), r, std::forward<Args>(args)..., false, false, true, false));
    }

    // conj matrix - scalar*conj matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    scal_gemm_return_type<T3, T1, T2> matmul(const conj_type<T1> &l, const scalconj_type<T3, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(r.left()), l.obj(), r.right().obj(), std::forward<Args>(args)..., false, false, true, true));
    }

    // scalar*conj matrix - conj matrix product
    template <typename T1, typename T2, typename T3, typename... Args>
    scal_gemm_return_type<T3, T1, T2> matmul(const scalconj_type<T3, T1> &l, const conj_type<T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.left()), l.right().obj(), r.obj(), std::forward<Args>(args)..., false, false, true, true));
    }

    // scalar matrix - scalar*conj matrix product
    template <typename T1, typename T2, typename T3, typename T4, typename... Args>
    scal_scal_gemm_return_type<T3, T1, T4, T2> matmul(const scal_type<T3, T1> &l, const scalconj_type<T4, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T3>(l.left()) * static_cast<T4>(r.left())), l.right(), r.right().obj(), std::forward<Args>(args)..., false, false, false, true));
    }

    // scalar*conj matrix - scalar matrix product
    template <typename T1, typename T2, typename T3, typename T4, typename... Args>
    scal_scal_gemm_return_type<T3, T1, T4, T2> matmul(const scalconj_type<T3, T1> &l, const scal_type<T4, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T3>(l.left()) * static_cast<T4>(r.left())), l.right().obj(), r.obj(), std::forward<Args>(args)..., false, false, true, false));
    }

    // trans matrix - scalar*conj matrix product
    template <typename T1, typename T2, typename T3, bool conjugate, typename... Args>
    scal_gemm_return_type<T3, T1, T2> matmul(const trans_type<T1, conjugate> &l, const scalconj_type<T3, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T3>(r.left()) * l.coeff()), l.matrix(), r.right().obj(), std::forward<Args>(args)..., true, false, conjugate, true));
    }

    // scalar*conj matrix - trans matrix product
    template <typename T1, typename T2, typename T3, bool conjugate, typename... Args>
    scal_gemm_return_type<T3, T1, T2> matmul(const scalconj_type<T3, T1> &l, const trans_type<T2, conjugate> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T3>(l.left()) * r.coeff()), l.right().obj(), r.matrix(), std::forward<Args>(args)..., false, true, true, conjugate));
    }

    // scalar*conj matrix - scalar*conj matrix product
    template <typename T1, typename T2, typename T3, typename T4, typename... Args>
    scal_scal_gemm_return_type<T3, T1, T4, T2> matmul(const scalconj_type<T3, T1> &l, const scalconj_type<T4, T2> &r, Args &&...args)
    {
        static_assert(sizeof...(Args) <= 1, "Failed to instantiate instance of matmul function.  The variadic template for matmul must have equal to or fewer than 1 elements");
        using rettype = gemm_type<T1, T2>;
        using value_type = typename rettype::value_type;
        CALL_AND_RETHROW(return rettype(static_cast<value_type>(static_cast<T3>(l.left()) * static_cast<T4>(r.left())), l.right(), r.right().obj(), std::forward<Args>(args)..., false, false, true, true));
    }

    template <typename T1, typename T2, typename B>
    auto matmul(const diagonal_matrix<T1, B> &a, const diagonal_matrix<T2, B> &b) -> decltype(hadamard(a, b)) { return hadamard(a, b); }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OVERLOADS OF OPERATOR* for the matrix - matrix and matrix - vector products
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // general matrix type -  product
    template <typename T1, typename T2>
    auto operator*(const T1 &l, const T2 &r) -> decltype(matmul(l, r)) { CALL_AND_RETHROW(return matmul(l, r)); }

} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_OVERLOADS_MATRIX_MATRIX_PRODUCT_HPP_//
