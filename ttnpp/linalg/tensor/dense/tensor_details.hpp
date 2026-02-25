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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_DETAILS_HPP_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_DETAILS_HPP_

#include "../../linalg_forward_decl.hpp"
#include "../../linalg_traits.hpp"

namespace linalg
{

    /**
     *  @cond INTERNAL
     *  Forward declaration of the Tensor class
     */
    // additional details for dense tensor objects providing additional functionality
    template <class ArrRef, size_t D = traits<ArrRef>::rank, bool is_mutable = traits<ArrRef>::is_mutable, typename backend_type = typename traits<ArrRef>::backend_type>
    class tensor_details
    {
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //                            GENERIC DETAILS OBJECTS FOR THE TENSORS                           //
    //////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename ArrType, typename backend>
    class tensor_details<ArrType, 1, false, backend>
    {
    public:
        using array_type = typename traits<ArrType>::base_type;
        using size_type = typename traits<backend>::size_type;
        using value_type = typename traits<ArrType>::value_type;

        inline size_type length() const { return static_cast<const array_type *>(this)->shape(0); }
        inline size_type incx() const { return 1; }
    };

    // The general matrix type.
    template <typename ArrType, typename backend>
    class tensor_details<ArrType, 2, false, backend>
    {
    public:
        using array_type = typename traits<ArrType>::base_type;
        using value_type = typename traits<ArrType>::value_type;
        using size_type = typename traits<backend>::size_type;
        inline size_type nrows() const { return static_cast<const array_type *>(this)->shape(0); }
        inline size_type ncols() const { return static_cast<const array_type *>(this)->shape(1); }

        size_type incx() const { return 1; }
        size_type diagonal_stride() const { return static_cast<const array_type *>(this)->shape(1) + 1; }
    };

    template <typename ArrType, typename backend>
    class tensor_details<ArrType, 3, false, backend>
    {
    public:
        using array_type = typename traits<ArrType>::base_type;
        using value_type = typename traits<ArrType>::value_type;
        using size_type = typename traits<backend>::size_type;
        inline size_type nslices() const { return static_cast<const array_type *>(this)->shape(0); }
        inline size_type nrows() const { return static_cast<const array_type *>(this)->shape(1); }
        inline size_type ncols() const { return static_cast<const array_type *>(this)->shape(2); }

        size_type incx() const { return 1; }
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //                          DETAILS OBJECTS FOR THE BLAS BACKEND TENSORS                        //
    //////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename ArrType, typename backend>
    class tensor_details<ArrType, 1, true, backend>
    {
    public:
        using array_type = typename traits<ArrType>::base_type;
        using size_type = typename traits<backend>::size_type;
        using value_type = typename traits<ArrType>::value_type;

        inline size_type length() const { return static_cast<const array_type *>(this)->shape(0); }
        inline size_type incx() const { return 1; }

        template <typename Func, typename... Args>
        void fill(Func &&f, Args &&...args)
        {
            array_type &a = static_cast<array_type &>(*this);
            size_type m = a.shape(0);
            backend::func_fill_1(a.buffer(), m, std::forward<Func>(f), std::forward<Args>(args)...);
        }
    };

    // The general matrix type.
    template <typename ArrType, typename backend>
    class tensor_details<ArrType, 2, true, backend>
    {
    public:
        using array_type = typename traits<ArrType>::base_type;
        using value_type = typename traits<ArrType>::value_type;
        using size_type = typename traits<backend>::size_type;
        inline size_type nrows() const { return static_cast<const array_type *>(this)->shape(0); }
        inline size_type ncols() const { return static_cast<const array_type *>(this)->shape(1); }
        size_type incx() const { return 1; }
        size_type diagonal_stride() const { return static_cast<const array_type *>(this)->shape(1) + 1; }

        template <typename Func, typename... Args>
        void fill(Func &&f, Args &&...args)
        {
            array_type &a = static_cast<array_type &>(*this);

            size_type m = a.shape(0);
            size_type n = a.shape(1);
            backend::func_fill_2(a.buffer(), m, n, std::forward<Func>(f), std::forward<Args>(args)...);
        }
        void set_subblock(const ArrType &block)
        {
            array_type &a = static_cast<array_type &>(*this);
            ASSERT(block.shape(0) <= a.shape(0) && block.shape(1) <= a.shape(1), "Failed to set subblock.  Block is larger than matrix.");
            CALL_AND_RETHROW(backend::copy_matrix_subblock(block.shape(0), block.shape(1), block.buffer(), block.shape(1), a.buffer(), a.shape(1)));
        }
    };

    template <typename ArrType, typename backend>
    class tensor_details<ArrType, 3, true, backend>
    {
    public:
        using array_type = typename traits<ArrType>::base_type;
        using value_type = typename traits<ArrType>::value_type;
        using size_type = typename traits<backend>::size_type;
        inline size_type nslices() const { return static_cast<const array_type *>(this)->shape(0); }
        inline size_type nrows() const { return static_cast<const array_type *>(this)->shape(1); }
        inline size_type ncols() const { return static_cast<const array_type *>(this)->shape(2); }
        size_type incx() const { return 1; }

        template <typename Func, typename... Args>
        void fill(Func &&f, Args &&...args)
        {
            array_type &a = static_cast<array_type &>(*this);
            size_type m = a.shape(0);
            size_type n = a.shape(1);
            size_type o = a.shape(2);
            backend::func_fill_3(a.buffer(), m, n, o, std::forward<Func>(f), std::forward<Args>(args)...);
        }
    };
    ///@endcond
} // namespace linalg //

#endif // PYTTN_LINALG_TENSOR_DENSE_TENSOR_DETAILS_HPP_//
