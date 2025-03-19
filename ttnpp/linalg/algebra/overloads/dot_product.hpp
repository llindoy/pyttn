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

#ifndef PYTTN_LINALG_ALGEBRA_OVERLOADS_DOT_PRODUCT_HPP_
#define PYTTN_LINALG_ALGEBRA_OVERLOADS_DOT_PRODUCT_HPP_

namespace linalg
{

    namespace internal
    {
        template <typename T1, typename T2, typename = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value, void>::type>
        struct is_valid_dot_product : public std::conditional<
                                          is_dense_tensor<T1>::value && is_dense_tensor<T2>::value,
                                          typename std::conditional<
                                              traits<T1>::rank == 1 && traits<T2>::rank == 1 &&
                                                  compatible_traits<T1, T2>::value,
                                              std::true_type,
                                              std::false_type>::type,
                                          typename std::conditional<
                                              is_diagonal_matrix_type<T1>::value && is_diagonal_matrix_type<T2>::value &&
                                                  compatible_traits<T1, T2>::value,
                                              std::true_type,
                                              std::false_type>::type>::type
        {
        };
    } // namespace internal

    // norm of an array
    template <typename T, typename = typename std::enable_if<is_linalg_object<T>::value, void>::type>
    typename linalg::get_real_type<typename traits<T>::value_type>::type abs(const T &a)
    {
        using value_type = typename std::remove_cv<typename traits<T>::value_type>::type;
        using backend_type = typename traits<T>::backend_type;

        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(true, a.size(), a.buffer(), a.incx(), a.buffer(), a.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");

        return std::sqrt(linalg::real(val));
    }

    // dot product of two arrays
    template <typename T1, typename T2>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const T1 &a, const T2 &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;

        ASSERT(a.size() == b.size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(false, a.size(), a.buffer(), a.incx(), b.buffer(), b.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return val;
    }

    // dot product of conjugate expressions
    template <typename T1, typename T2>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const conj_type<T1> &a, const T2 &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;

        ASSERT(a.obj().size() == b.size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(true, b.size(), a.obj().buffer(), a.obj().incx(), b.buffer(), b.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return val;
    }

    template <typename T1, typename T2>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const T1 &a, const conj_type<T2> &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;

        ASSERT(a.size() == b.obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(true, a.size(), b.obj().buffer(), b.obj().incx(), a.buffer(), a.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return val;
    }

    template <typename T1, typename T2>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const conj_type<T1> &a, const conj_type<T2> &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;
        using std::conj;

        ASSERT(a.obj().size() == b.obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(false, a.obj().size(), a.obj().buffer(), a.obj().incx(), b.obj().buffer(), b.obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return conj(val);
    }

    // dot product of scalar times conjugate types
    template <typename T1, typename T2, typename T3>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type() * T3())>::type dot_product(const scalconj_type<T3, T1> &a, const T2 &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;

        ASSERT(a.right().obj().size() == b.size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(true, b.size(), a.right().obj().buffer(), a.right().obj().incx(), b.buffer(), b.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return val * a.left()();
    }

    template <typename T1, typename T2, typename T3>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type() * T3())>::type dot_product(const T1 &a, const scalconj_type<T3, T2> &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;

        ASSERT(a.size() == b.right().obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(false, a.size(), b.right().obj().buffer(), b.right().obj().incx(), a.buffer(), a.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return val * b.left()();
    }

    template <typename T1, typename T2, typename T3, typename T4>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type() * T3() * T4())>::type dot_product(const scalconj_type<T3, T1> &a, const scalconj_type<T4, T2> &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;
        using std::conj;

        ASSERT(a.right().obj().size() == b.right().obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(false, a.right().obj().size(), a.right().obj().buffer(), a.right().obj().incx(), b.right().obj().buffer(), b.right().obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return conj(val) * a.left()() * b.right()();
    }

    // dot product of scalar*conj and conjugate types
    template <typename T1, typename T2, typename T3>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type() * T3())>::type dot_product(const scalconj_type<T3, T1> &a, const conj_type<T2> &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;
        using std::conj;

        ASSERT(a.right().obj().size() == b.obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(false, b.obj().size(), a.right().obj().buffer(), a.right().obj().incx(), b.obj().buffer(), b.obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return conj(val) * a.left()();
    }

    template <typename T1, typename T2, typename T3>
    typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type() * T3())>::type dot_product(const conj_type<T1> &a, const scalconj_type<T3, T2> &b)
    {
        using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
        using backend_type = typename traits<T1>::backend_type;
        using std::conj;

        ASSERT(a.obj().size() == b.right().obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::dot(false, a.obj().size(), a.obj().buffer(), a.obj().incx(), b.right().obj().buffer(), b.right().obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
        return conj(val) * b.left()();
    }

} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_OVERLOADS_DOT_PRODUCT_HPP_//
