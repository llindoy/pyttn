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

#ifndef PYTTN_TTNS_LIB_SOP_SOP_TREE_HPP_
#define PYTTN_TTNS_LIB_SOP_SOP_TREE_HPP_

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

#include "autoSOP_node.hpp"
#include "sSOP.hpp"

namespace ttns
{

    template <typename T>
    class sop_node_data;

    template <typename T>
    class sop_term
    {
    public:
        using real_type = typename tmp::get_real_type<T>::type;

        using tree_type = tree<sop_node_data<T>>;
        using node_type = typename tree_type::node_type;

        using accum_coeff_type = std::vector<T>;

        using spf_index_type = std::vector<std::vector<std::array<size_t, 2>>>;
        using mf_index_type = std::vector<mf_index<size_t>>;

        template <typename Y, typename V>
        friend class operator_container;

    public:
        sop_term() {}
        sop_term(const sop_term &o) = default;
        sop_term(sop_term &&o) = default;
        sop_term &operator=(const sop_term &o) = default;

        sop_term(const operator_contraction_info<T> &o)
        {
            CALL_AND_HANDLE(m_oci = o, "Failed to construct operator contraction term from operator_contraction_info.");
        }

        void set_operator_contraction_info(const operator_contraction_info<T> &o)
        {
            CALL_AND_HANDLE(m_oci = o, "Failed to set operator_contraction_info.");
        }

        void clear()
        {
            CALL_AND_HANDLE(m_oci.clear(), "Failed to clear the coefficient array.");
            CALL_AND_HANDLE(m_mf.clear(), "Failed to clear mf matrix object.");
            CALL_AND_HANDLE(m_spf.clear(), "Failed to clear spf matrix object.");
        }

        const sSOP<T> &spf() const { return m_spf; }
        sSOP<T> &spf() { return m_spf; }

        const sSOP<T> &mf() const { return m_mf; }
        sSOP<T> &mf() { return m_mf; }

        bool is_identity_spf() const { return m_oci.is_identity_spf(); }
        bool is_identity_mf() const { return m_oci.is_identity_mf(); }

        const accum_coeff_type &accum_coeff() const { return m_oci.accum_coeff(); }
        const T &accum_coeff(size_t i) const { return m_oci.accum_coeff(i); }

        const T &coeff() const { return m_oci.coeff(); }

        const spf_index_type &spf_indexing() const { return m_oci.spf_indexing(); }
        const mf_index_type &mf_indexing() const { return m_oci.mf_indexing(); }

        size_t nspf_terms() const { return m_oci.nspf_terms(); }
        size_t nmf_terms() const { return m_oci.nmf_terms(); }

        const operator_contraction_info<T> &contraction_info() const { return m_oci; }
#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void serialize(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("spf", m_spf)), "Failed to serialise operator term object.  Error when serialising the spf matrix.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("mf", m_mf)), "Failed to serialise operator term object.  Error when serialising the mf matrix.");

            CALL_AND_HANDLE(ar(cereal::make_nvp("contraction_info", m_oci)), "Failed to serialise operator term object.  Error when serialising the contraction info.");
        }
#endif

    protected:
        sSOP<T> m_spf;
        sSOP<T> m_mf;

        operator_contraction_info<T> m_oci;
    };

    template <typename T>
    class sop_node_data
    {
    public:
        using real_type = typename tmp::get_real_type<T>::type;

        template <typename Y, typename V>
        friend class operator_container;

    public:
        sop_node_data() {}

        sop_node_data(const sop_node_data &o) = default;
        sop_node_data(sop_node_data &&o) = default;
        sop_node_data &operator=(const sop_node_data &o) = default;
        sop_node_data &operator=(sop_node_data &&o) = default;

        void set_contraction_info(const std::vector<operator_contraction_info<T>> &o)
        {
            m_term.clear();
            m_term.resize(o.size());
            for (size_t i = 0; i < o.size(); ++i)
            {
                m_term[i] = o[i];
            }
        }

        ~sop_node_data() {}

        void clear()
        {
            try
            {
                for (size_t i = 0; i < m_term.size(); ++i)
                {
                    m_term[i].clear();
                }
                m_term.clear();
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to clear operator node object.");
            }
        }

        size_t nterms() const { return m_term.size(); }

        const sop_term<T> &term(size_t i) const
        {
            ASSERT(i < m_term.size(), "Index out of bounds.");
            return m_term[i];
        }

        sop_term<T> &term(size_t i)
        {
            ASSERT(i < m_term.size(), "Index out of bounds.");
            return m_term[i];
        }

        const sop_term<T> &operator[](size_t i) const { return m_term[i]; }
        sop_term<T> &operator[](size_t i) { return m_term[i]; }

#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void serialize(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("terms", m_term)), "Failed to serialise operator node object.  Error when serialising the terms.");
        }
#endif

    protected:
        std::vector<sop_term<T>> m_term;
    }; // sop_node_data

    namespace node_data_traits
    {
        // clear traits for the operator node data object
        template <typename T>
        struct clear_traits<sop_node_data<T>>
        {
            void operator()(sop_node_data<T> &t) { CALL_AND_RETHROW(t.clear()); }
        };

    } // namespace node_data_traits
} // namespace ttns

#endif // PYTTN_TTNS_LIB_SOP_SOP_TREE_HPP_
