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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_BLAS_EXEC_CONTEXT_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_BLAS_EXEC_CONTEXT_HPP_

#include <cstdint>
#include <cstdlib>
#include <common/exception_handling.hpp>
#include "../../backends/blas/blas_backend.hpp"
#include "../execProps.hpp"
#include "execContext.hpp"

namespace linalg
{
    namespace memory
    {

        /* A class for handling information about device information for operators acting on linalg objects.*/
        template <>
        class ExecContext<blas_backend>
        {
        protected:
            const ExecDomain<blas_backend>& m_domain;   //MPI rank local domain
            std::size_t m_nthreads; 
            std::size_t m_id;
        
        public:
            ExecContext(const ExecDomain<blas_backend>& domain, std::size_t nthreads = 1, std::size_t id = 0) : m_domain(domain), m_nthreads(nthreads), m_id(id) {}
            ExecContext(const ExecContext& o) = default;
            ExecContext(ExecContext&& o) = default;

            const ExecDomain<blas_backend>& domain() const {return m_domain;}
            std::size_t nthreads() const {return m_nthreads;}
            std::size_t& nthreads() {return m_nthreads;}
            std::size_t id() const{return m_id;}
            std::size_t& id(){return m_id;}
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_BLAS_EXEC_CONTEXT_HPP_
