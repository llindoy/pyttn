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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_BLAS_EXEC_DOMAIN_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_BLAS_EXEC_DOMAIN_HPP_

#include <cstdint>
#include <cstdlib>
#include <common/exception_handling.hpp>
#include "../../backends/blas/blas_backend.hpp"
#include "../execProps.hpp"

namespace linalg
{
    namespace memory
    {

        /* A class for handling information about device information for a linalg object.*/
        template <>
        class ExecDomain<blas_backend>
        {
        protected:
            int m_mpi_rank;     //flags the mpi rank.  This is purely metadata. 
        
        public:
            ExecDomain() : m_mpi_rank(0) {}
            ExecDomain(int mpi_rank) : m_mpi_rank(mpi_rank){}
            ExecDomain(const ExecDomain& o) = default;
            ExecDomain(ExecDomain&& o) = default;

            bool operator==(const ExecDomain& o){return true;}
            int mpi_rank() const{return m_mpi_rank;}
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_BLAS_EXEC_DOMAIN_HPP_
