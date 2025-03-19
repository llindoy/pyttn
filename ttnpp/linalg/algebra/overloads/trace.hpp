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

#ifndef PYTTN_LINALG_ALGEBRA_OVERLOADS_TRACE_HPP_
#define PYTTN_LINALG_ALGEBRA_OVERLOADS_TRACE_HPP_

namespace linalg
{

    namespace internal
    {
        template <typename T>
        struct is_valid_trace : public std::conditional<
                                    is_dense_tensor<T>::value,
                                    typename std::conditional<
                                        traits<T>::rank == 2 && !is_expression<T>::value,
                                        std::true_type,
                                        std::false_type>::type,
                                    typename std::conditional<
                                        is_diagonal_matrix_type<T>::value && !is_expression<T>::value,
                                        std::true_type,
                                        std::false_type>::type>::type
        {
        };
    } // namespace internal

    // dot product of two arrays
    template <typename T>
    typename std::enable_if<internal::is_valid_trace<T>::value, typename traits<T>::value_type>::type trace(const T &a)
    {
        using value_type = typename std::remove_cv<typename traits<T>::value_type>::type;
        using backend_type = typename traits<T>::backend_type;

        ASSERT(a.shape(0) == a.shape(1), "Failed to evaluate trace of matrix.  The matrix is not a square matrix.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::trace(a.shape(1), a.buffer(), a.diagonal_stride()), "Failed to evaluate trace of matrix.  Failed to call the backend::trace routine.");
        return val;
    }

    // dot product of conjugate expressions
    template <typename T>
    typename std::enable_if<internal::is_valid_trace<T>::value, typename traits<T>::value_type>::type trace(const conj_type<T> &a)
    {
        using value_type = typename std::remove_cv<typename traits<T>::value_type>::type;
        using backend_type = typename traits<T>::backend_type;

        ASSERT(a.shape(0) == a.shape(1), "Failed to evaluate trace of matrix.  The matrix is not a square matrix.");
        value_type val;
        CALL_AND_HANDLE(val = backend_type::trace(a.shape(1), a.buffer(), a.diagonal_stride()), "Failed to evaluate trace of matrix.  Failed to call the backend::trace routine.");
        return conj(val);
    }
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_OVERLOADS_TRACE_HPP_//
