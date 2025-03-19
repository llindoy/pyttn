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

#ifndef PYTTN_LINALG_LINALG_TRAITS_HPP_
#define PYTTN_LINALG_LINALG_TRAITS_HPP_

#include "linalg_forward_decl.hpp"

namespace linalg
{

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //                         traits objects for the dense tensor types                           //
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct traits<T, validate_value_type<T>>
    {
        using value_type = void;
        using backend_type = void;
        using base_type = void;
        using size_type = void;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = false;
        static constexpr size_t rank = 0;
    };

    /*
    template <typename T>
    struct traits<T, typename std::enable_if<not is_linalg_object<typename remove_cvref<T>::type>::value, void>::type >
    {
        using value_type = void;
        using backend_type = void;
        using base_type = void;
        using size_type = void;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = false;
        static constexpr size_t rank = 0;
    };
    */

    template <typename T, typename backend>
    struct traits<expression_templates::literal_type<T, backend>, validate_value_type<T>>
    {
        using value_type = T;
        using backend_type = backend;
        using base_type = expression_templates::literal_type<T, backend>;
        using size_type = typename backend::size_type;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = false;
        static constexpr size_t rank = 0;
    };

    template <typename T, size_t D, typename backend>
    struct traits<tensor<T, D, backend>, validate_backend_type<backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend::size_type;
        using base_type = tensor_base<tensor<T, D, backend>>;
        using container_type = tensor<T, D, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = true;
        static constexpr size_type rank = D;
    };

    template <typename ArrType, typename data_type, size_t D>
    struct traits<tensor_slice<ArrType, data_type, D>, void>
    {
        using backend_type = typename traits<ArrType>::backend_type;
        using value_type = data_type;
        using size_type = typename backend_type::size_type;
        using base_type = tensor_slice_base<tensor_slice<ArrType, data_type, D>>;
        using container_type = ArrType;

        static constexpr bool is_mutable = (!std::is_const<data_type>::value) && traits<ArrType>::is_mutable;
        static constexpr bool is_resizable = false;
        static constexpr size_type rank = D;
    };

    template <typename T, size_t D, typename backend>
    struct traits<tensor_view<T, D, backend>, validate_backend_type<backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend_type::size_type;
        using base_type = tensor_view_base<tensor_view<T, D, backend>>;
        using container_type = tensor_view<T, D, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = false;
        static constexpr size_type rank = D;
    };

    template <typename T, size_t D, typename backend>
    struct traits<reinterpreted_tensor<T, D, backend>, validate_backend_type<backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend_type::size_type;
        using base_type = tensor_view<T, D, backend>;
        using container_type = reinterpreted_tensor<T, D, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = false;
        static constexpr size_type rank = D;
    };

    template <typename T, typename backend>
    struct traits<hermitian_matrix<T, backend>, validate_value_backend_type<T, backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend_type::size_type;
        using base_type = tensor_view<T, 2, backend>;
        using container_type = hermitian_matrix<T, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = false;
        static constexpr size_type rank = 2;
    };

    template <typename T, typename backend>
    struct traits<upper_hessenberg_matrix<T, backend>, validate_value_backend_type<T, backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend_type::size_type;
        using base_type = tensor_view<T, 2, backend>;
        using container_type = upper_hessenberg_matrix<T, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = false;
        static constexpr size_type rank = 2;
    };

    // traits types for the crtp base types.  These default to using whatever implementation is provided
    template <typename T>
    struct traits<tensor_base<T>>
    {
        using impl_traits = traits<T>;
        using value_type = typename impl_traits::value_type;
        using backend_type = typename impl_traits::backend_type;
        using size_type = typename impl_traits::size_type;
        using container_type = typename impl_traits::container_type;

        static constexpr bool is_mutable = impl_traits::is_mutable;
        static constexpr bool is_resizable = impl_traits::is_resizable;
        static constexpr size_type rank = impl_traits::rank;
    };

    template <typename T>
    struct traits<tensor_view_base<T>>
    {
        using impl_traits = traits<T>;
        using value_type = typename impl_traits::value_type;
        using backend_type = typename impl_traits::backend_type;
        using size_type = typename impl_traits::size_type;
        using container_type = typename impl_traits::container_type;

        static constexpr bool is_mutable = impl_traits::is_mutable;
        static constexpr bool is_resizable = impl_traits::is_resizable;
        static constexpr size_type rank = impl_traits::rank;
    };

    template <typename T>
    struct traits<tensor_slice_base<T>>
    {
        using impl_traits = traits<T>;
        using value_type = typename impl_traits::value_type;
        using backend_type = typename impl_traits::backend_type;
        using size_type = typename impl_traits::size_type;
        using container_type = typename impl_traits::container_type;

        static constexpr bool is_mutable = impl_traits::is_mutable;
        static constexpr bool is_resizable = impl_traits::is_resizable;
        static constexpr size_type rank = impl_traits::rank;
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //                        traits objects for the sparse tensor types                           //
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename backend>
    struct traits<csr_matrix<T, backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend::size_type;
        using index_type = typename backend::index_type;
        using base_type = csr_matrix_base<csr_matrix<T, backend>>;
        using container_type = csr_matrix<T, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = true;
        static constexpr size_type rank = 2;
    };

    /*

    template <typename T, typename backend>
    struct traits<bcsr_matrix<T, backend> >
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend::size_type;
        using real_type = bcsr_matrix<typename get_real_type<T>::type, backend>;
        using base_type = bcsr_matrix_base<bcsr_matrix<T, backend> >;

        static constexpr bool is_mutable(){return true;}
        static constexpr size_type rank(){return 2;}
    };
    */

    template <typename T, typename backend>
    struct traits<diagonal_matrix<T, backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend::size_type;
        using base_type = diagonal_matrix_base<diagonal_matrix<T, backend>>;
        using container_type = diagonal_matrix<T, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = true;
        static constexpr size_type rank = 2;
    };

    template <typename T, typename backend>
    struct traits<symmetric_tridiagonal_matrix<T, backend>>
    {
        using value_type = T;
        using backend_type = backend;
        using size_type = typename backend::size_type;
        using real_type = symmetric_tridiagonal_matrix<typename get_real_type<T>::type, backend>;
        using base_type = symmetric_tridiagonal_matrix_base<symmetric_tridiagonal_matrix<T, backend>>;
        using container_type = symmetric_tridiagonal_matrix<T, backend>;

        static constexpr bool is_mutable = !std::is_const<T>::value;
        static constexpr bool is_resizable = true;
        static constexpr size_type rank = 2;
    };

    template <typename T>
    struct traits<csr_matrix_base<T>>
    {
        using impl_traits = traits<T>;
        using value_type = typename impl_traits::value_type;
        using backend_type = typename impl_traits::backend_type;
        using size_type = typename impl_traits::size_type;
        using container_type = typename impl_traits::container_type;

        static constexpr bool is_mutable = impl_traits::is_mutable;
        static constexpr bool is_resizable = impl_traits::is_resizable;
        static constexpr size_type rank = impl_traits::rank;
    };

    template <typename T>
    struct traits<diagonal_matrix_base<T>>
    {
        using impl_traits = traits<T>;
        using value_type = typename impl_traits::value_type;
        using backend_type = typename impl_traits::backend_type;
        using size_type = typename impl_traits::size_type;
        using container_type = typename impl_traits::container_type;

        static constexpr bool is_mutable = impl_traits::is_mutable;
        static constexpr bool is_resizable = impl_traits::is_resizable;
        static constexpr size_type rank = impl_traits::rank;
    };
    template <typename T>
    struct traits<symmetric_tridiagonal_matrix_base<T>>
    {
        using impl_traits = traits<T>;
        using value_type = typename impl_traits::value_type;
        using backend_type = typename impl_traits::backend_type;
        using size_type = typename impl_traits::size_type;
        using container_type = typename impl_traits::container_type;

        static constexpr bool is_mutable = impl_traits::is_mutable;
        static constexpr bool is_resizable = impl_traits::is_resizable;
        static constexpr size_type rank = impl_traits::rank;
    };

    template <typename T>
    struct traits<special_matrix_base<T>>
    {
        using impl_traits = traits<T>;
        using value_type = typename impl_traits::value_type;
        using backend_type = typename impl_traits::backend_type;
        using size_type = typename impl_traits::size_type;
        using container_type = typename impl_traits::container_type;

        static constexpr bool is_mutable = impl_traits::is_mutable;
        static constexpr bool is_resizable = impl_traits::is_resizable;
        static constexpr size_type rank = impl_traits::rank;
    };

} // namespace linalg

#endif // PYTTN_LINALG_LINALG_TRAITS_HPP_//
