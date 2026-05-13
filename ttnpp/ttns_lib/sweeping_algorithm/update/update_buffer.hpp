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

#ifndef PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_UPDATE_BUFFER_HPP_
#define PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_UPDATE_BUFFER_HPP_

#include "../../ttn/ms_ttn.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/vector.hpp>
#endif

namespace ttns
{

  template <typename T, typename backend>
  class multiset_update_buffer
  {
    using size_type = typename linalg::traits<backend>::size_type;
    using ttn_type = ms_ttn<T, backend>;

protected:
    linalg::vector<T, backend> m_Abuf;
    linalg::vector<T, backend> m_resbuf;
    std::vector<linalg::matrix<T, backend>> m_res;
    size_type m_maxcapacity;
    std::vector<size_t> m_set_capacity;

  public:
    template <typename buftype>
    void setup(const std::vector<buftype> &A)
    {
        if (m_res.size() != A.size())
        {
            m_res.resize(A.size());
        }
        size_t size = 0;
        for (size_t i = 0; i < m_res.size(); ++i)
        {
            m_res[i].resize(A[i].size(0), A[i].size(1));
            size += A[i].size();
        }
        m_Abuf.resize(size);
        CALL_AND_HANDLE(ttn_type::flatten(A, m_Abuf), "Failed to flatten buffer.");
        m_resbuf.resize(size);
    }

    void unpack_results()
    {
      CALL_AND_HANDLE(ttn_type::unpack(m_resbuf, m_res),
                      "Failed to unpack buffer.");
    }

    void initialise(const ms_ttn<T, backend> &A)
    {
        clear();

        size_type maxcapacity = 0;
        m_set_capacity = std::vector<size_t>(A.nset(), 0);
        for (const auto &a : A)
        {
            size_type capacity = a.buffer_maxcapacity();
            if (capacity > maxcapacity)
            {
                maxcapacity = capacity;
            }
            for (size_t i = 0; i < A.nset(); ++i)
            {
                size_type scap = a()[i].capacity();
                if (scap > m_set_capacity[i])
                {
                    m_set_capacity[i] = scap;
                }
            }
        }

        m_maxcapacity = maxcapacity;
        CALL_AND_HANDLE(m_Abuf.reallocate(maxcapacity),
                        "Failed to reserve storage for internal types.");
        CALL_AND_HANDLE(m_resbuf.reallocate(maxcapacity),
                        "Failed to reserve storage for internal types.");
        m_res.resize(A.nset());
        for (size_t i = 0; i < A.nset(); ++i)
        {
            m_res[i].reallocate(m_set_capacity[i]);
        }
    }

    void clear()
    {
        m_Abuf.clear();
        m_resbuf.clear();
        for (size_t i = 0; i < m_res.size(); ++i)
        {
            m_res[i].clear();
        }
        m_res.clear();
    }

    linalg::vector<T, backend> &A() { return m_Abuf; }
    const linalg::vector<T, backend> &A() const { return m_Abuf; }
    linalg::vector<T, backend> &resbuf() { return m_resbuf; }
    const linalg::vector<T, backend> &resbuf() const { return m_resbuf; }
    std::vector<linalg::matrix<T, backend>> &res() { return m_res; }
    const std::vector<linalg::matrix<T, backend>> &re() const { return m_res; }


#ifdef CEREAL_LIBRARY_FOUND
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxcapacity", m_maxcapacity)), "Failed to serialise multiset update buffer.  Failed to serialise max capacity.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("set_capacity", m_set_capacity)), "Failed to serialise multiset update buffer.  Failed to serialise set capacity.");
    }
    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxcapacity", m_maxcapacity)), "Failed to serialise multiset update buffer.  Failed to serialise max capacity.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("set_capacity", m_set_capacity)), "Failed to serialise multiset update buffer.  Failed to serialise set capacity.");


        CALL_AND_HANDLE(m_Abuf.reallocate(m_maxcapacity),
                        "Failed to reserve storage for internal types.");
        CALL_AND_HANDLE(m_resbuf.reallocate(m_maxcapacity),
                        "Failed to reserve storage for internal types.");
        m_res.resize(m_set_capacity.size());
        for (size_t i = 0; i < m_set_capacity.size(); ++i)
        {
            m_res[i].reallocate(m_set_capacity[i]);
        }
    }
#endif

  };

} // namespace ttns

#endif // PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_UPDATE_BUFFER_HPP_
