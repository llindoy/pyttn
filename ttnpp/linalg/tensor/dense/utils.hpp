/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_PRINT_UTILS_HPP_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_PRINT_UTILS_HPP_

#include <type_traits>
#include <cstdlib>
#include <cstdint>

// TODO: Implement stl allocators (and potentially an aligned allocator) to handle memory rather than the hacky approach I have currently taken.
namespace linalg
{
    namespace internal
    {
        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_unsigned<Int>::value && std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return i < bounds; }

        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_integral<Int>::value && !std::is_unsigned<Int>::value && std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return (i >= 0 && static_cast<size_type>(i) < bounds); }

        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_integral<size_type>::value && std::is_unsigned<Int>::value && !std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return (i < static_cast<Int>(bounds) || bounds < 0); }

        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_integral<Int>::value && std::is_integral<size_type>::value && !std::is_unsigned<Int>::value && !std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return i < bounds; }       
    }

    template <size_t D>
    struct NDIndex
    {
        template <typename size_type, typename IntegerType, typename... Args>
        static inline size_type flatten(const std::array<size_type, D>& stride, IntegerType i, Args... args)
        {
            return i * stride[D - sizeof...(args) - 1] + flatten(stride, args...); 
        }
        template <typename size_type, typename IntegerType, typename... Args>
        static inline size_type flatten(const std::array<size_type, D>& /*stride*/, IntegerType i) { return i; }
        ///@cond INTERNAL - we might want to move this elsewhere - this should be common to all dense tensor types.
        // get the index in the array corresponding to the parameter pack.
        template <typename size_type, typename IntegerType, typename... Args>
        static inline size_type flatten_check(const std::array<size_type, D>& shape, const std::array<size_type, D>& stride, IntegerType i, Args... args)
        {
            ASSERT(internal::compare_bounds(i, shape[D - sizeof...(args) - 1]), "Unable to get flattened index.  One of the unflattened indices was out of bounds.");
            CALL_AND_HANDLE(return i * stride[D - sizeof...(args) - 1] + flatten_check(shape, stride,  args...), "Unable to get flattened index.  Error on iterated get_index call.");
        }
        template <typename size_type, typename IntegerType>
        static inline size_type get_index_bounds_check(const std::array<size_type, D>& shape, const std::array<size_type, D>& /*stride*/, IntegerType i)
        {
            ASSERT(internal::compare_bounds(i, shape[D - 1]), "Unable to get flattened index.  Final unflattened index was out of bounds.");
            return i;
        }
    };

    namespace io
    {
        /*
        ///////////////////////////////////////////////////////////////////////////////////////
        //            ostream operators for the D dimensional blas tensor objects            //
        ///////////////////////////////////////////////////////////////////////////////////////
        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<traits<array_type>::rank == 1 && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            os << "[";
            for (size_t i = 0; i < t.shape(0); ++i)
            {
                os << t(i) << (i + 1 == t.shape(0) ? "]" : ", ");
            }
            return os;
        }

        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<traits<array_type>::rank == 2 && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            os << "[";
            for (size_t i = 0; i < t.shape(0); ++i)
            {
                os << "[";
                for (size_t j = 0; j < t.shape(1); ++j)
                {
                    os << t(i, j) << (j + 1 == t.shape(1) ? "]" : ", ");
                }
                os << (i + 1 == t.shape(0) ? "]" : ",") << std::endl;
            }
            return os;
        }

        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<traits<array_type>::rank == 3 && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            os << "[";
            for (size_t i = 0; i < t.shape(0); ++i)
            {
                os << "[";
                for (size_t j = 0; j < t.shape(1); ++j)
                {
                    os << "[";
                    for (size_t k = 0; k < t.shape(2); ++k)
                    {
                        os << t(i, j, k) << (k + 1 == t.shape(2) ? "]" : ", ");
                    }
                    os << (j + 1 == t.shape(1) ? "]" : ",");
                }
                os << (i + 1 == t.shape(0) ? "]" : ",");
            }
            return os;
        }

        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<(traits<array_type>::rank > 3) && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            os << "shape: [";
            for (size_t i = 0; i < t.rank; ++i)
            {
                os << t.shape(i) << (i + 1 == t.rank ? "]" : ", ");
            }
            os << std::endl
            << "data : [";
            for (size_t i = 0; i < t.size(); ++i)
            {
                os << t(i) << (i + 1 == t.size() ? "]" : ", ");
            }
            return os;
        }

        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<traits<array_type>::rank == 1 && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            using std::abs;
            os << "[";
            for (size_t i = 0; i < t.shape(0); ++i)
            {
                os << t(i).real() << (t(i).imag() < 0.0 ? "-" : "+") << abs(t(i).imag()) << "i" << (i + 1 == t.shape(0) ? "]" : ", ");
            }
            return os;
        }

        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<traits<array_type>::rank == 2 && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            using std::abs;
            os << "[";
            for (size_t i = 0; i < t.shape(0); ++i)
            {
                os << "[";
                for (size_t j = 0; j < t.shape(1); ++j)
                {
                    os << t(i, j).real() << (t(i, j).imag() < 0.0 ? "-" : "+") << abs(t(i, j).imag()) << "i" << (j + 1 == t.shape(1) ? "]" : ", ");
                }
                os << (i + 1 == t.shape(0) ? "]" : ",") << std::endl;
            }
            return os;
        }

        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<traits<array_type>::rank == 3 && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            using std::abs;
            os << "[";
            for (size_t i = 0; i < t.shape(0); ++i)
            {
                os << "[";
                for (size_t j = 0; j < t.shape(1); ++j)
                {
                    os << "[";
                    for (size_t k = 0; k < t.shape(2); ++k)
                    {
                        os << t(i, j, k).real() << (t(i, j, k).imag() < 0.0 ? "-" : "+") << abs(t(i, j, k).imag()) << "i" << (k + 1 == t.shape(2) ? "]" : ", ");
                    }
                    os << (j + 1 == t.shape(1) ? "]" : ",");
                }
                os << (i + 1 == t.shape(0) ? "]" : ",");
            }
            return os;
        }

        template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
        typename std::enable_if<(traits<array_type>::rank > 3) && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
        {
            using std::abs;
            os << "shape: [";
            for (size_t i = 0; i < t.rank; ++i)
            {
                os << t.shape(i) << (i + 1 == t.rank ? "]" : ", ");
            }
            os << std::endl
            << "data : [";
            for (size_t i = 0; i < t.size(); ++i)
            {
                os << t(i).real() << (t(i).imag() < 0.0 ? "-" : "+") << abs(t(i).imag()) << "i" << (i + 1 == t.size() ? "]" : ", ");
            }
            return os;
        }
            */
    }
}

#endif // PYTTN_LINALG_TENSOR_DENSE_PRINT_UTILS_HPP_//
