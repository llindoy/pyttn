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

template <typename T>
inline void convert_to_dense(const sOP& op, const system_modes& sysinf, linalg::matrix<T>& ret)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    for(size_t i = 0; i < dims.size(); ++i)
    {
        std::cout << dims[i] << std::endl;
    }
    site_operator<T> _op(op, sysinf);
    CALL_AND_HANDLE(_op.todense(dims, ret), "Failed to convert sOP operator to dense matrix.");
}

template <typename T>
inline void convert_to_dense(const sPOP& op, const system_modes& sysinf, linalg::matrix<T>& ret)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    product_operator<T> _op(op, sysinf);
    CALL_AND_HANDLE(_op.todense(dims, ret), "Failed to convert sOP operator to dense matrix.");
}

template <typename T, typename U>
inline void convert_to_dense(const sNBO<T>& op, const system_modes& sysinf, linalg::matrix<U>& ret)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    product_operator<U> _op(op, sysinf);
    CALL_AND_HANDLE(_op.todense(dims, ret), "Failed to convert sOP operator to dense matrix.");
}

template <typename T, typename U>
inline void convert_to_dense(const sSOP<T>& op, const system_modes& sysinf, linalg::matrix<U>& ret)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    size_t nhilb = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        nhilb *= dims[i];
    }

    linalg::matrix<U> ret2(nhilb, nhilb);
    ret.resize(nhilb, nhilb);
    ret.fill_zeros();
    for (auto& t : op)
    {
        ret2.fill_zeros();
        product_operator<U> _op(t, sysinf);
        CALL_AND_HANDLE(_op.todense(dims, ret2), "Failed to convert sOP operator to dense matrix.");
        ret += ret2;

    }
}

template <typename T, typename U>
inline void convert_to_dense(const SOP<T>& op, const system_modes& sysinf, linalg::matrix<U>& ret)
{
    std::vector<size_t> dims;
    sysinf.get_mode_dimensions(dims);    
    size_t nhilb = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        nhilb *= dims[i];
    }

    linalg::matrix<U> ret2(nhilb, nhilb);
    ret.resize(nhilb, nhilb);
    ret.fill_zeros();
    for (const auto& t : op)
    {
        ret2.fill_zeros();
        product_operator<U> _op(expand_term(t, op.operator_dictionary()), sysinf);
        CALL_AND_HANDLE(_op.todense(dims, ret2), "Failed to convert sOP operator to dense matrix.");
        ret += ret2;
    }
}


}

#endif  //PYTTN_TTNS_LIB_SOP_TODENSE_HPP_
