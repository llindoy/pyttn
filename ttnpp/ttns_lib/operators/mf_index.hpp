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

#ifndef PYTTN_TTNS_LIB_OPERATORS_MF_INDEX_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_MF_INDEX_HPP_

#include <linalg/linalg.hpp>
#include <common/exception_handling.hpp>

#include <memory>
#include <list>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <list>
#include <tuple>
#include <memory>
#include <utility>
#include <initializer_list>
#include <type_traits>

namespace ttns
{
    template <typename size_type>
    class mf_index
    {
    public:
        using container_type = std::vector<std::array<size_type, 2>>;

    public:
        mf_index() {}
        mf_index(size_type s)
        {
            CALL_AND_HANDLE(resize(s), "Failed to construct mf_index object.");
        }

        mf_index(const mf_index &o) = default;
        mf_index(mf_index &&o) = default;
        mf_index(size_type parentindex, const container_type &v) : m_parent_index(parentindex), m_sibling_indices(v) {}
        mf_index(size_type parentindex, container_type &&v) : m_parent_index(parentindex), m_sibling_indices(std::move(v)) {}

        mf_index &operator=(const mf_index &o) = default;
        mf_index &operator=(mf_index &&o) = default;

        void clear()
        {
            m_parent_index = 0;
            m_sibling_indices.clear();
        }

        void resize(size_type size)
        {
            CALL_AND_HANDLE(m_sibling_indices.resize(size), "Failed to resize sibling indices.");
        }

        const size_type &parent_index() const { return m_parent_index; }
        size_type &parent_index() { return m_parent_index; }

        const std::array<size_type, 2> &sibling_index(size_type i) const { return m_sibling_indices[i]; }
        std::array<size_type, 2> &sibling_index(size_type i) { return m_sibling_indices[i]; }

        const container_type &sibling_indices() const { return m_sibling_indices; }

#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void serialize(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("parent_index", m_parent_index)), "Failed to serialise mf_index object.  Error when serialising the parent_index.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("sibling_indices", m_sibling_indices)), "Failed to serialise mf_index object.  Error when serialising the sibling_indices.");
        }
#endif
    protected:
        size_type m_parent_index;
        container_type m_sibling_indices;
    };

    template <typename I>
    std::ostream &operator<<(std::ostream &os, const mf_index<I> &t)
    {
        os << "p: " << t.parent_index() << " s: ";
        for (size_t i = 0; i < t.sibling_indices().size(); ++i)
        {
            os << "(" << t.sibling_index(i)[0] << ", " << t.sibling_index(i)[1] << ")";
        }
        return os;
    }

} // namespace ttns

#endif // PYTTN_TTNS_LIB_OPERATORS_MF_INDEX_HPP_
