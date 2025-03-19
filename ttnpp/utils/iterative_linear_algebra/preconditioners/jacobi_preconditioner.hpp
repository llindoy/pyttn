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

#ifndef PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_PRECONDITIONRS_JACOBI_PRECONDITIONER_HPP_
#define PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_PRECONDITIONRS_JACOBI_PRECONDITIONER_HPP_

#include "preconditioner_base.hpp"
#include <common/exception_handling.hpp>

#include <linalg/linalg.hpp>

namespace utils
{

    namespace preconditioner
    {

        template <typename T, typename backend>
        class jacobi : public preconditioner<T, backend>
        {
        public:
            jacobi() {}
            jacobi(const linalg::diagonal_matrix<T, backend> &mat)
                : m_mat(mat), m_temp(mat.size(0)) {}
            jacobi(const jacobi &o) = default;
            jacobi(jacobi &&o) = default;

            jacobi &operator=(const jacobi &o) = default;
            jacobi &operator=(jacobi &&o) = default;

            template <typename Vin>
            void apply(Vin &x)
            {
                ASSERT(x.size() == m_temp.size(),
                       "Failed to apply preconditioner.  Invalid shape of input array.");
                auto xi = x.reinterpret_shape(m_temp.shape());
                CALL_AND_HANDLE(m_temp = m_mat * xi,
                                "Failed to apply the preconditioner matrix.")
                CALL_AND_HANDLE(
                    x = m_temp,
                    "Failed to copy the preconditioned result into the return location.");
            }

            void clear()
            {
                m_mat.clear();
                m_temp.clear();
            }

            void initialise(const linalg::diagonal_matrix<T, backend> &mat)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to copy preconditioning matrix.");
                CALL_AND_HANDLE(m_temp.resize(m_mat.size(0)),
                                "Failed to resize working buffer.");
            }

        protected:
            linalg::diagonal_matrix<T, backend> m_mat;
            linalg::vector<T, backend> m_temp;
        };

    } // namespace preconditioner
} // namespace utils

#endif // PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_PRECONDITIONRS_JACOBI_PRECONDITIONER_HPP_
