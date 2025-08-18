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

#ifndef PYTTN_TTNS_LIB_TTN_TTN_SCALAR_OP_HPP_
#define PYTTN_TTNS_LIB_TTN_TTN_SCALAR_OP_HPP_

#include <linalg/linalg.hpp>
#include <functional>


#include "ttn.hpp"
#include "multiset_ttn_slice.hpp"


namespace ttns
{

template <typename T, typename backend>
class ttn_scalar_op
{
public:
    using result_type = ttn<T, backend>;
    using ttn_item_type = std::pair<T, std::reference_wrapper<const result_type>>;

    //add support for slice terms in here
    //using slice_item_type = std::pair<T, std::reference_wrapper<ms_ttn<T, backend>>>;

    ttn_scalar_op(){}
    ttn_scalar_op(const result_type & A, const result_type& B)
    {
        set_sum(T(1), A, T(1), B);
    }

    void set_sum(const T& vA, const result_type & A, const T& vB, const result_type& B)
    {
        m_ttn_terms.resize(2);
        m_ttn_terms[0] = std::make_pair(vA, std::cref(A));
        m_ttn_terms[1] = std::make_pair(vB, std::cref(B));
    }

    void add_term(const T& v, const result_type& A)
    {
        m_ttn_terms.push_back(std::make_pair(v, std::cref(A)));
    }
protected:
    std::vector<item_type> m_ttn_terms;
};

}   //namespace ttn

#endif //PYTTN_TTNS_LIB_TTN_TTN_SCALAR_OP_HPP_