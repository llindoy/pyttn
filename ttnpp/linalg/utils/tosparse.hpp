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

#ifndef PYTTN_LINALG_UTILS_TOSPARSE_HPP_
#define PYTTN_LINALG_UTILS_TOSPARSE_HPP_


#include "../linalg_forward_decl.hpp"
#include "../linalg_traits.hpp"

namespace linalg
{

template <typename T>
class tosparse
{
protected:
    using real_type = typename linalg::get_real_type<T>::type;

public:
    static inline void convert(const linalg::matrix<T>& in, linalg::csr_matrix<T>& out, real_type tol=1e-14)
    {
        //determine the number of non-zero elements in the dense matrix
        size_t nnz = 0;
        for(size_t i = 0; i < in.size(0); ++i)
        {
            for(size_t j = 0; j < in.size(1); ++j)
            {
                if(linalg::abs(in(i, j)) >= tol)
                {
                    ++nnz;
                }
            }
        }

        out.resize(nnz, in.size(0), in.size(1));
        auto buffer = out.buffer();
        auto rowptr = out.rowptr();
        auto colind = out.colind();

        rowptr[0] = 0;
        nnz = 0;
        for(size_t i = 0; i < in.size(0); ++i)
        {
            for(size_t j = 0; j < in.size(1); ++j)
            {
                if(linalg::abs(in(i, j)) >= tol)
                {
                    buffer[nnz] = in(i, j);
                    colind[nnz] = j;
                    ++nnz;
                }
            }
            rowptr[i+1] = nnz;
        }
    }
};

}

#endif //PYTTN_LINALG_UTILS_TOSPARSE_HPP_