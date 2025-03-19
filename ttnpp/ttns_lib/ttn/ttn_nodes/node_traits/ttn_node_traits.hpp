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

#ifndef PYTTN_TTNS_LIB_TTN_TTN_NODES_NODE_TRAITS_TTN_NODE_TRAITS_HPP_
#define PYTTN_TTNS_LIB_TTN_TTN_NODES_NODE_TRAITS_TTN_NODE_TRAITS_HPP_

#include <linalg/linalg.hpp>

#include <type_traits>

namespace ttns
{

    namespace node_data_traits
    {
        // assignment traits for the ttn node data object
        template <typename T, typename U, typename backend1, typename backend2>
        struct assignment_traits<ttn_node_data<T, backend1>, ttn_node_data<U, backend2>>
        {
            using hdata = ttn_node_data<T, backend1>;
            using hdata2 = ttn_node_data<U, backend2>;
            using is_applicable = std::is_convertible<U, T>;

            inline typename std::enable_if<std::is_convertible<U, T>::value, void>::type operator()(hdata &o, const hdata2 &i)
            {
                CALL_AND_RETHROW(o = i);
            }
        };

        // assignment traits for the ttn node data object
        template <typename T, typename U, typename backend1, typename backend2>
        struct assignment_traits<ttn_node_data<T, backend1>, linalg::matrix<U, backend2>>
        {
            using hdata = ttn_node_data<T, backend1>;
            using mat = linalg::matrix<U, backend2>;

            using is_applicable = std::is_convertible<U, T>;

            inline typename std::enable_if<std::is_convertible<U, T>::value, void>::type operator()(hdata &o, const mat &i)
            {
                CALL_AND_RETHROW(o = i);
            }
        };

        // resize traits for the ttn node data object
        template <typename T, typename U, typename backend1, typename backend2>
        struct resize_traits<ttn_node_data<T, backend1>, ttn_node_data<U, backend2>>
        {
            template <typename V, typename backend>
            using hdata = ttn_node_data<V, backend>;

            using is_applicable = std::true_type;

            inline void operator()(hdata<T, backend1> &o, const hdata<U, backend2> &i)
            {
                CALL_AND_RETHROW(o.resize(i.hrank(), i.dims()));
            }
        };

        // resize traits for the ttn node data object
        template <typename T, typename U, typename backend1, typename backend2>
        struct reallocate_traits<ttn_node_data<T, backend1>, ttn_node_data<U, backend2>>
        {
            template <typename V, typename backend>
            using hdata = ttn_node_data<V, backend>;

            using is_applicable = std::true_type;

            inline void operator()(hdata<T, backend1> &o, const hdata<U, backend2> &i)
            {
                CALL_AND_RETHROW(o.reallocate(i.max_hrank(), i.max_dims()));
            }
        };

        // size comparison traits for the ttn node data object
        template <typename T, typename U, typename backend1, typename backend2>
        struct size_comparison_traits<ttn_node_data<T, backend1>, ttn_node_data<U, backend2>>
        {
            template <typename V, typename backend>
            using hdata = ttn_node_data<V, backend>;

            using is_applicable = std::true_type;

            inline bool operator()(const hdata<T, backend1> &o, const hdata<U, backend2> &i)
            {
                if (o.nmodes() != i.nmodes())
                {
                    return false;
                }
                else
                {
                    return ((o.shape() == i.shape()) && (o.dims() == i.dims()));
                }
            }
        };

        template <typename T, typename U, typename backend1, typename backend2>
        struct size_comparison_traits<ttn_node_data<T, backend1>, linalg::matrix<U, backend2>>
        {
            using hdata = ttn_node_data<T, backend1>;
            using mat = linalg::matrix<U, backend2>;

            using is_applicable = std::true_type;

            inline bool operator()(const hdata &o, const mat &i)
            {
                return (o.shape() == i.shape());
            }
        };

        // clear traits for the ttn node data object
        template <typename T, typename backend>
        struct clear_traits<ttn_node_data<T, backend>>
        {
            void operator()(ttn_node_data<T, backend> &t) { CALL_AND_RETHROW(t.clear()); }
        };
    }

}

#endif // PYTTN_TTNS_LIB_TTN_TTN_NODES_NODE_TRAITS_TTN_NODE_TRAITS_HPP_//
