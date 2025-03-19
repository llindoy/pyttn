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

#ifndef PYTTN_TTNS_LIB_SOP_MULTISET_SOP_HPP_
#define PYTTN_TTNS_LIB_SOP_MULTISET_SOP_HPP_

#include "SOP.hpp"

#include <tuple>
#include <vector>
#include <algorithm>

namespace ttns
{

    template <typename T>
    class multiset_SOP;
    template <typename T>
    std::ostream &operator<<(std::ostream &os, const multiset_SOP<T> &op);

    // the string sum of product operator class used for storing the representation of the Hamiltonian of interest.
    template <typename T>
    class multiset_SOP
    {
    public:
        using operator_dictionary_type = std::vector<std::vector<std::string>>;
        using container_type = std::map<std::pair<size_t, size_t>, SOP<T>>;

        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;

    public:
        multiset_SOP() : m_nset(1), m_nmodes(0) {}
        multiset_SOP(size_t nset, size_t nmodes) : m_nset(nset), m_nmodes(nmodes) {}
        multiset_SOP(size_t nset, size_t nmodes, const std::string &label) : m_nset(nset), m_nmodes(nmodes), m_label(label) {}

        multiset_SOP(const multiset_SOP &o) = default;
        multiset_SOP(multiset_SOP &&o) = default;

        multiset_SOP &operator=(const multiset_SOP &o) = default;
        multiset_SOP &operator=(multiset_SOP &&o) = default;

        void resize(size_t nset, size_t nmodes)
        {
            for (auto &it : m_terms)
            {
                it.second.clear();
            }
            m_terms.clear();
            m_nset = nset;
            m_nmodes = nmodes;
        }
        void clear()
        {
            for (auto &it : m_terms)
            {
                it.second.clear();
            }
            m_terms.clear();
            m_label.clear();
        }

        friend std::ostream &operator<< <T>(std::ostream &os, const multiset_SOP<T> &op);

        size_t nset() const { return m_nset; }
        size_t nterms() const { return m_terms.size(); }
        size_t nmodes() const { return m_nmodes; }

        const std::string &label() const { return m_label; }
        std::string &label() { return m_label; }

        SOP<T> &operator()(size_t i, size_t j)
        {
            ASSERT(i < m_nset && j < m_nset, "Failed to access sop.  Index out of bounds.");
            std::pair<size_t, size_t> key = std::make_pair(i, j);

            auto iter = m_terms.find(key);
            if (iter == m_terms.end())
            {
                iter = m_terms.insert({key, SOP<T>(m_nmodes)}).first;
            }
            return iter->second;
        }

        const SOP<T> &operator()(size_t i, size_t j) const
        {
            ASSERT(i < m_nset && j < m_nset, "Failed to access sop.  Index out of bounds.");
            std::pair<size_t, size_t> key = std::make_pair(i, j);
            auto iter = m_terms.find(key);
            ASSERT(iter != m_terms.end(), "Failed to access term.  No such term has been bound.");
            return iter->second;
        }

        template <typename... Args>
        void insert(size_t i, size_t j, Args &&...args)
        {
            ASSERT(i < m_nset && j < m_nset, "Failed to access sop.  Index out of bounds.");
            std::pair<size_t, size_t> key = std::make_pair(i, j);

            auto iter = m_terms.find(key);
            if (iter == m_terms.end())
            {
                iter = m_terms.insert({key, SOP<T>(m_nmodes)}).first;
            }
            iter->second.insert(std::forward<Args>(args)...);
        }

        void set(size_t i, size_t j, const SOP<T> &sop)
        {
            ASSERT(i < m_nset && j < m_nset, "Failed to access sop.  Index out of bounds.");
            std::pair<size_t, size_t> key = std::make_pair(i, j);

            auto iter = m_terms.find(key);
            if (iter == m_terms.end())
            {
                iter = m_terms.insert({key, SOP<T>(m_nmodes)}).first;
            }
            iter->second = sop;
        }

    protected:
        container_type m_terms;
        size_t m_nset;
        size_t m_nmodes;
        std::string m_label;

    public:
        inline bool set_is_fermionic_mode(std::vector<bool> &is_fermion_mode) const
        {
            bool res = true;
            for (const auto &it : m_terms)
            {
                res = res && it.second.set_is_fermionic_mode(is_fermion_mode);
            }
            return res;
        }

        multiset_SOP &jordan_wigner(const system_modes &sys_info, double tol = 1e-15)
        {
            for (auto &it : m_terms)
            {
                it.second.jordan_wigner(sys_info, tol);
            }
            return *this;
        }

        iterator begin() { return iterator(m_terms.begin()); }
        iterator end() { return iterator(m_terms.end()); }
        const_iterator begin() const { return const_iterator(m_terms.begin()); }
        const_iterator end() const { return const_iterator(m_terms.end()); }
    };

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const ttns::multiset_SOP<T> &op)
    {
        for (const auto &o : op)
        {
            os << "set index: " << std::get<0>(o.first) << " " << std::get<1>(o.first) << std::endl;
            os << o.second << std::endl;
        }
        return os;
    }
}

#endif // PYTTN_TTNS_LIB_SOP_MULTISET_SOP_HPP_
