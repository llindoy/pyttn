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

#ifndef PYTTN_UTILS_PRODUCT_ITERATOR_HPP_
#define PYTTN_UTILS_PRODUCT_ITERATOR_HPP_

#include <functional>
#include <vector>
#include <common/exception_handling.hpp>

namespace utils
{
    template <typename T>
    class product_iterator
    {
    protected:
        using vector_type = std::vector<T>;
        using vector_iterator = typename vector_type::iterator;
        using iter_vec = std::vector<vector_iterator>;
        using ref_vec = std::vector<std::reference_wrapper<T>>;
        using pointer_vec = std::vector<T *>;

        using self_type = product_iterator<T>;

    public:
        product_iterator(iter_vec &state, const iter_vec &begin, const iter_vec &end) : m_state(state), m_begin(begin), m_end(end)
        {
            ASSERT(state.size() == begin.size() && begin.size() == end.size(), "Cannot construct this iterator.");
        }

        bool operator==(const self_type &other) const { return m_state == other.m_state; }
        bool operator!=(const self_type &other) const { return m_state != other.m_state; }

        ref_vec operator*() const
        {
            ref_vec ret;
            for (auto v : m_state)
            {
                ret.push_back(std::reference_wrapper<T>(*v));
            }
            return ret;
        }

        T &operator[](size_t i)
        {
            ASSERT(i < m_state.size(), "Cannot access element.  Index out of bounds.");
            return *(m_state[i]);
        }

        iter_vec state() { return m_state; }

        self_type &operator++()
        {
            for (size_t i = 0; i < m_state.size(); ++i)
            {
                size_t ind = m_state.size() - i - 1;
                if (ind != 0 && m_state[ind] + 1 == m_end[ind])
                {
                    m_state[ind] = m_begin[ind];
                }
                else if (ind == 0 && m_state[0] + 1 == m_end[0])
                {
                    m_state = m_end;
                }
                else
                {
                    ++m_state[ind];
                    return *this;
                }
            }
            return *this;
        }

        self_type operator++(int)
        {
            self_type ret(*this);
            ++(*this);
            return ret;
        }

        size_t size() const { return m_state.size(); }

    protected:
        iter_vec m_state;
        iter_vec m_begin;
        iter_vec m_end;
    };

    template <typename T>
    product_iterator<T> prod_begin(std::vector<std::vector<T>> &v)
    {
        using iterator = typename std::vector<T>::iterator;
        std::vector<iterator> state(v.size());
        std::vector<iterator> begin(v.size());
        std::vector<iterator> end(v.size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            state[i] = v[i].begin();
            begin[i] = v[i].begin();
            end[i] = v[i].end();
        }
        return product_iterator<T>(state, begin, end);
    }

    template <typename T>
    product_iterator<T> prod_end(std::vector<std::vector<T>> &v)
    {
        using iterator = typename std::vector<T>::iterator;
        std::vector<iterator> state(v.size());
        std::vector<iterator> begin(v.size());
        std::vector<iterator> end(v.size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            state[i] = v[i].end();
            begin[i] = v[i].begin();
            end[i] = v[i].end();
        }
        return product_iterator<T>(state, begin, end);
    }
}

#endif // PYTTN_UTILS_PRODUCT_ITERATOR_HPP_
