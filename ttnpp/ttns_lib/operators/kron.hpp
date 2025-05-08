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

#ifndef PYTTN_TTNS_LIB_OPERATORS_KRON_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_KRON_HPP_

#include <vector>
#include <linalg/linalg.hpp>

namespace ttns
{

    class kron
    {
    protected:
        static inline void advance_state(const std::vector<size_t>& dims, std::vector<size_t>& mind)
        {
            size_t Nm = mind.size();
            for(size_t i = 0; i < Nm; ++i)
            {
                ++mind[Nm-(i+1)];
                if(mind[Nm-(i+1)] >= dims[Nm-(i+1)])
                {
                    mind[Nm-(i+1)] = 0;
                }
                else
                {
                    return;
                }
            }
        }

    public:
        template <typename T>
        static inline void eval(const std::vector<linalg::matrix<T>> &x, linalg::matrix<T> &r)
        {
            std::vector<size_t> mdims(x.size());
            std::vector<size_t> mind(x.size());     std::fill(mind.begin(), mind.end(), 0);
            std::vector<size_t> ndims(x.size());
            std::vector<size_t> nind(x.size());     std::fill(nind.begin(), nind.end(), 0);
            size_t mdim=1;
            size_t ndim=1;
            for (size_t i = 0; i < x.size(); ++i)
            {
                mdims[i] = x[i].shape(0);
                ndims[i] = x[i].shape(1);
                mdim *= mdims[i];
                ndim *= ndims[i];
            }

            CALL_AND_HANDLE(r.resize(mdim, ndim), "Failed to resize result matrix for Kron");

            //now actually perform the Kron
            for(size_t m = 0; m < mdim; ++m)
            {
                std::fill(nind.begin(), nind.end(), 0);
                for (size_t n = 0; n < ndim; ++n)
                {
                    T coeff(1.0);
                    for(size_t k=0; k < x.size(); ++k)
                    {
                        coeff *= x[k](mind[k], nind[k]);
                    }
                    r(m, n)=coeff;
                    advance_state(ndims, nind);
                }
                advance_state(mdims, mind);
            }
        }
    };

} // namespace ttns
#endif // PYTTN_TTNS_LIB_OPERATORS_KRON_HPP_//
