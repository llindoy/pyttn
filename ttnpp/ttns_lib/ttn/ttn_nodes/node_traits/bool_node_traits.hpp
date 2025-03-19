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

#ifndef PYTTN_TTNS_LIB_TTN_TTN_NODES_NODE_TRAITS_BOOL_NODE_TRAITS_HPP_
#define PYTTN_TTNS_LIB_TTN_TTN_NODES_NODE_TRAITS_BOOL_NODE_TRAITS_HPP_

namespace ttns
{

    namespace node_data_traits
    {
        // bool nodes
        template <>
        struct assignment_traits<bool, bool>
        {
            using is_applicable = std::true_type;

            inline void operator()(bool &o, const bool &i) { o = i; }
        };

        // resize traits for tensor and matrix objects
        template <typename... Args>
        struct resize_traits<bool, Args...>
        {
            using is_applicable = std::true_type;
            inline void operator()(bool & /* o */, const Args &.../* args */) {}
        };

        template <typename... Args>
        struct reallocate_traits<bool, Args...>
        {
            using is_applicable = std::true_type;
            inline void operator()(bool & /* o */, const Args &.../* args */) {}
        };

        // size comparison traits for the tensor and matrix objects
        template <typename... Args>
        struct size_comparison_traits<bool, Args...>
        {
            using is_applicable = std::true_type;

            inline bool operator()(const bool & /* o */, const Args &.../* i */) { return true; }
        };

        /*
         * size_t nodes
         */
        template <>
        struct default_initialisation_traits<size_t>
        {
            using is_applicable = std::true_type;
            template <typename... Args>
            void operator()(size_t &n, Args &&.../* args */) { n = 0; }
        };

        template <>
        struct assignment_traits<size_t, size_t>
        {
            using is_applicable = std::true_type;

            inline void operator()(size_t &o, const size_t &i) { o = i; }
        };

        template <typename... Args>
        struct resize_traits<size_t, Args...>
        {
            using is_applicable = std::true_type;
            inline void operator()(size_t &o, const Args &.../* args */) { o = 0; }
        };

        template <typename... Args>
        struct reallocate_traits<size_t, Args...>
        {
            using is_applicable = std::true_type;
            inline void operator()(size_t &o, const Args &.../* args */) { o = 0; }
        };

        template <typename... Args>
        struct size_comparison_traits<size_t, Args...>
        {
            using is_applicable = std::true_type;

            inline size_t operator()(const size_t & /* o */, const Args &.../* i */) { return true; }
        };
    }

}

#endif // PYTTN_TTNS_LIB_TTN_TTN_NODES_NODE_TRAITS_BOOL_NODE_TRAITS_HPP_//
