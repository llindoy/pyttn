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

            ExecDomain& operator=(const ExecDomain& o) = default;
            ExecDomain& operator=(ExecDomain&& o) = default;

            bool operator==(const ExecDomain& o){return true}

            const int& mpi_rank() const{return m_mpi_rank;}
            int& mpi_rank(){return m_mpi_rank;}
        };

        /* A class for handling information about device information for operators acting on linalg objects.*/
        template <>
        class ExecContext<blas_backend>
        {
        protected:
            const ExecDomain<blas_backend>* m_domain;   //MPI rank local domain
            std::size_t m_nthreads; 
            std::size_t m_id;
        
        public:
            ExecContext(const ExecDomain<blas_backend>& domain, std::size_t nthreads = 1, std::size_t id = 0) : m_domain(&domain), m_nthreads(nthreads), m_id(id) {}
            ExecContext(const ExecContext& o) = default;
            ExecContext(ExecContext&& o) = default;
            ExecContext& operator=(const ExecContext& o) = default;
            ExecContext& operator=(ExecContext&& o) = default;

            const ExecDomain<blas_backend>& domain() const {return *m_domain;}
            std::size_t nthreads() const {return m_nthreads;}
            std::size_t& nthreads() {return m_nthreads;}
            std::size_t id() const{return m_id;}
            std::size_t& id(){return m_id;}
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_BLAS_EXEC_DOMAIN_HPP_
