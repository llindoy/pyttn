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

#ifndef PYTTN_TTNS_LIB_SOP_TODENSE_HPP_
#define PYTTN_TTNS_LIB_SOP_TODENSE_HPP_

#include "../operators/site_operators/site_operator.hpp"
#include "../operators/product_operator.hpp"
#include "SOP.hpp"
#include "sSOP.hpp"
#include "system_information.hpp"

namespace ttns
{

template <typename T, typename backend>
inline void toDense(const site_operator<T>& op, const system_modes& sysinf)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    CALL_AND_HANDLE(return op.todense(dims), "Failed to convert site operator to dense matrix.");
}

template <typename T, typename backend>
inline void toDense(const product_operator<T>& op, const system_modes& sysinf)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    CALL_AND_HANDLE(return op.todense(dims), "Failed to convert product operator to dense matrix.");
}

/*
template <typename T>
inline void toDense(const sOP& op, const system_modes& sysinf)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    site_operator<T> _op(op, system_modes);
    CALL_AND_HANDLE(return _op.todense(dims), "Failed to convert sOP operator to dense matrix.");
}

template <typename T>
inline void toDense(const sPOP& op, const system_modes& sysinf)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    product_operator<T> _op(op, system_modes);
    CALL_AND_HANDLE(return _op.todense(dims), "Failed to convert sOP operator to dense matrix.");
}
*/

template <typename T>
inline void toDense(const sNBO<T>& op, const system_modes& sysinf)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    product_operator<T> _op(op, system_modes);
    CALL_AND_HANDLE(return _op.todense(dims), "Failed to convert sOP operator to dense matrix.");
}

template <typename T>
inline void toDense(const sSOP<T>& op, const system_modes& sysinf)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    size_t nhilb = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        nhilb *= dims[i];
    }

    linalg::matrix<T> ret(nhilb, nhilb);
    ret.fill_zeros();
    for (auto& t : op)
    {
        product_operator<T> _op(op, system_modes);
        CALL_AND_HANDLE(ret += _op.todense(dims), "Failed to convert sOP operator to dense matrix.");
    }
    return ret;
}

template <typename T>
inline void toDense(const SOP<T>& op, const system_modes& sysinf)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    size_t nhilb = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        nhilb *= dims[i];
    }

    linalg::matrix<T> ret(nhilb, nhilb);
    ret.fill_zeros();
    for (size_t i = 0; i < op.nterms(); ++i)
    {
        product_operator<T> _op(op.expand_term(i), system_modes);
        CALL_AND_HANDLE(ret += _op.todense(dims), "Failed to convert sOP operator to dense matrix.");
    }
    return ret;
}


}

#endif  //PYTTN_TTNS_LIB_SOP_TODENSE_HPP_
