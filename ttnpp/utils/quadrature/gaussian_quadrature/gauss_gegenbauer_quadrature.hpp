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

#ifndef PYTTN_UTILS_QUADRATURE_GAUSSIAN_QUADRATURE_GAUSS_GEGENBAUER_QUADRATURE_HPP_
#define PYTTN_UTILS_QUADRATURE_GAUSSIAN_QUADRATURE_GAUSS_GEGENBAUER_QUADRATURE_HPP_

#include "gauss_jacobi_quadrature.hpp"
#include <common/exception_handling.hpp>

namespace utils
{
    namespace quad
    {
        namespace gauss
        {

            template <typename T, typename backend = linalg::blas_backend>
            class gegenbauer : public jacobi<T, backend>
            {
            public:
                gegenbauer(size_t N, T alpha, bool m = true)
                    : jacobi<T, backend>(N, alpha, alpha, m) {}

                gegenbauer(const gegenbauer &o) = default;
                gegenbauer(gegenbauer &&o) = default;

                gegenbauer &operator=(const gegenbauer &o) = default;
                gegenbauer &operator=(gegenbauer &&o) = default;
            };

        } // namespace gauss
    } // namespace quad
} // namespace utils

#endif // PYTTN_UTILS_QUADRATURE_GAUSSIAN_QUADRATURE_GAUSS_GEGENBAUER_QUADRATURE_HPP_
