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

#ifndef PYTTN_TTNS_LIB_CORE_OBSERVABLE_NODE_HPP_
#define PYTTN_TTNS_LIB_CORE_OBSERVABLE_NODE_HPP_

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

#include "../ttn/tree/tree_node.hpp"

namespace ttns
{

    template <typename T, typename B>
    class observable_node_data
    {
    public:
        using backend_type = B;
        using size_type = typename backend_type::size_type;
        using real_type = typename tmp::get_real_type<T>::type;

        observable_node_data() {}
        observable_node_data(const observable_node_data &o) = default;
        observable_node_data(observable_node_data &&o) = default;

        ~observable_node_data() {}

        observable_node_data &operator=(const observable_node_data &o) = default;
        observable_node_data &operator=(observable_node_data &&o) = default;

        void resize(size_type r)
        {
            m_data.resize(r);
        }

        void expand_buffer(size_type size)
        {
            ASSERT(m_data.size() > 0, "Failed to expand buffers as the buffer has not been allocated.");
            m_data.resize(size, m_data[0]);
        }

        template <typename be>
        typename std::enable_if<not std::is_same<be, backend_type>::value, observable_node_data &>::type operator=(const observable_node_data<T, be> &o)
        {
            try
            {
                for (size_type i = 0; i < o.size(); ++i)
                {
                    m_data[i] = o[i];
                }
                m_has_identity = o.has_identity();
                m_id = o.id();
                return *this;
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to copy assign operator term.");
            }
        }

        void resize_matrices(size_type n, size_type m)
        {
            m_dim1 = n;
            m_dim2 = m;
            for (size_type i = 0; i < m_data.size(); ++i)
            {
                CALL_AND_HANDLE(m_data[i].resize(n, m), "Failed to setup matrices for observable node.");
            }
            if (m_has_identity)
            {
                CALL_AND_HANDLE(m_id.resize(n, m), "Failed to resize identity buffer.");
            }
        }

        void reallocate_matrices(size_type capacity)
        {
            if (capacity > m_capacity)
            {
                m_capacity = capacity;
                for (size_type i = 0; i < m_data.size(); ++i)
                {
                    CALL_AND_HANDLE(m_data[i].reallocate(capacity), "Failed to setup matrices for observable node.");
                }
                if (m_has_identity)
                {
                    CALL_AND_HANDLE(m_id.reallocate(capacity), "Failed to reserve identtiy buffer.");
                }
            }
        }

        void store_identity()
        {
            m_has_identity = true;
            if (m_id.capacity() < m_capacity)
            {
                m_id.reallocate(m_capacity);
            }
            m_id.resize(m_dim1, m_dim2);
        }

        void clear()
        {
            try
            {
                for (size_type i = 0; i < m_data.size(); ++i)
                {
                    m_data[i].clear();
                }
                m_data.clear();
                m_id.clear();
                m_capacity = 0;
                m_dim1 = 0;
                m_dim2 = 0;
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to clear observable node object.");
            }
        }

        size_t matrix_capacity() const { return m_capacity; }
        std::array<size_t, 2> matrix_size() const
        {
            std::array<size_t, 2> ret;
            ret[0] = m_dim1;
            ret[1] = m_dim2;
            return ret;
        }

        size_type size() const { return m_data.size(); }

        const linalg::matrix<T, B> &operator[](size_type i) const { return m_data[i]; }
        linalg::matrix<T, B> &operator[](size_type i) { return m_data[i]; }

        const linalg::matrix<T, B> &spf(size_type i) const { return m_data[i]; }
        linalg::matrix<T, B> &spf(size_type i) { return m_data[i]; }

        const linalg::matrix<T, B> &id() const { return m_id; }
        linalg::matrix<T, B> &id()
        {
            m_has_identity = true;
            return m_id;
        }

        const linalg::matrix<T, B> &spf_id() const { return m_id; }
        linalg::matrix<T, B> &spf_id()
        {
            m_has_identity = true;
            return m_id;
        }

        bool has_identity() const { return m_has_identity; }
        bool &has_identity() { return m_has_identity; }

#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void serialize(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("data", m_data)), "Failed to serialise observable node object.  Error when serialising the terms.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("id", m_id)), "Failed to serialise observable node object.  Error when serialising the terms.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("has_id", m_has_identity)), "Failed to serialise observable node object.  Error when serialising the terms.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("capacity", m_capacity)), "Failed to serialise observable node object.  Error when serialising the terms.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("dim1", m_dim1)), "Failed to serialise observable node object.  Error when serialising the terms.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("dim2", m_dim2)), "Failed to serialise observable node object.  Error when serialising the terms.");
        }
#endif

    protected:
        std::vector<linalg::matrix<T, B>> m_data;
        linalg::matrix<T, B> m_id;
        bool m_has_identity = false;
        size_type m_capacity = 0;
        size_type m_dim1 = 0;
        size_type m_dim2 = 0;
    };

    namespace node_data_traits
    {
        // assignment traits for the tensor and matrix objects
        template <typename T, typename U, typename be1, typename be2>
        struct assignment_traits<observable_node_data<T, be1>, observable_node_data<U, be2>>
        {
            using is_applicable = std::is_convertible<U, T>;

            inline void operator()(observable_node_data<T, be1> &o, const observable_node_data<U, be2> &i) { CALL_AND_RETHROW(o = i); }
        };

        // resize traits for tensor and matrix objects
        template <typename T, typename U, typename be1, typename be2>
        struct resize_traits<observable_node_data<T, be1>, observable_node_data<U, be2>>
        {
            using is_applicable = std::true_type;
            inline void operator()(observable_node_data<T, be1> &o, const observable_node_data<U, be2> &i) { CALL_AND_RETHROW(o.resize(i.size())); }
        };

        template <typename T, typename U, typename be1, typename be2>
        struct reallocate_traits<observable_node_data<T, be1>, observable_node_data<U, be2>>
        {
            using is_applicable = std::true_type;
            inline void operator()(observable_node_data<T, be1> &o, const observable_node_data<U, be2> &i) { CALL_AND_RETHROW(o.reserve(i.size())); }
        };

        // size comparison traits for the tensor and matrix objects
        template <typename T, typename U, typename be1, typename be2>
        struct size_comparison_traits<observable_node_data<T, be1>, observable_node_data<U, be2>>
        {
            using is_applicable = std::true_type;

            inline bool operator()(const observable_node_data<T, be1> &o, const observable_node_data<U, be2> &i) { return o.size() == i.size(); }
        };

        template <typename T, typename be>
        struct clear_traits<observable_node_data<T, be>>
        {
            void operator()(observable_node_data<T, be> &t) { CALL_AND_RETHROW(t.clear()); }
        };

    } // namespace node_data_traits

    template <typename optype>
    using ms_opnode_data = std::vector<std::vector<optype>>;

    template <typename T, typename B>
    using ms_observable_node_data = ms_opnode_data<observable_node_data<T, B>>;

    template <typename optype>
    class ms_observable_slice
    {
        using obj_type = tree_node<ms_opnode_data<optype>> &;

    public:
        ms_observable_slice() = delete;
        ms_observable_slice(obj_type o, size_t i, size_t c) : m_obj(o), m_i(i), m_c(c) {}

        optype &operator()() const { return m_obj()[m_i][m_c]; }

        ms_observable_slice parent() const { return ms_observable_slice(m_obj.parent(), m_i, m_c); }
        ms_observable_slice operator[](size_t i) const { return ms_observable_slice(m_obj[i], m_i, m_c); }

    protected:
        obj_type m_obj;
        size_t m_i;
        size_t m_c;
    };

} // namespace ttns

#endif // PYTTN_TTNS_LIB_CORE_OBSERVABLE_NODE_HPP_
