/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
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

#ifndef PYTTN_LINALG_UTILS_ALLOCATOR_UTILS_HPP_
#define PYTTN_LINALG_UTILS_ALLOCATOR_UTILS_HPP_

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <complex>
#include "../backends/blas/blas_backend.hpp"

namespace linalg
{
    namespace memory
    {
        template <typename backend>
        class ExecDomain;


        /* A class for handling information about device information for a linalg object.*/
        template <>
        class ExecDomain<blas_backend>
        {
        protected:
            int m_mpi_rank;
        
        public:
            ExecDomain() : m_mpi_rank(0) {}
            ExecDomain(int mpi_rank) : m_mpi_rank(mpi_rank){}
            ExecDomain(const ExecDomain& o) = default;
            ExecDomain(ExecDomain&& o) = default;

            ExecDomain& operator=(const ExecDomain& o) = default;
            ExecDomain& operator=(ExecDomain&& o) = default;

            bool operator==(const ExecDomain& o)
            {
                //if the Domains have the same rank and if (gpu_id is less than 0 check that the other is less than zero (i.e. both cpu devices).  Otherwise check that they are both on the same gpu).
                return m_mpi_rank == o.m_mpi_rank;
            }

            const int& mpi_rank() const{return m_mpi_rank;}
            int& mpi_rank(){return m_mpi_rank;}
        };

        template <typename T, typename backend>
        struct alignment_for_type
        {
            std::size_t eval(const ExecDomain<backend>& domain)
            {
                return std::alignof(T);
            }
        };
    }
}

#endif //PYTTN_LINALG_UTILS_ALLOCATOR_HPP_
