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

#ifndef PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_PRECONDITIONRS_PRECONDITIONER_BASE_HPP_
#define PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_PRECONDITIONRS_PRECONDITIONER_BASE_HPP_

namespace utils
{

    namespace preconditioner
    {

        template <typename T, typename backend>
        class preconditioner
        {
        };

        template <typename T, typename backend>
        class identity : public preconditioner<T, backend>
        {
        public:
            identity() {}
            identity(const identity &o) = default;
            identity(identity &&o) = default;

            identity &operator=(const identity &o) = default;
            identity &operator=(identity &&o) = default;

            template <typename Vin>
            void apply(Vin &) {}
            void clear() {}
            void initialise() {}
        };

    } // namespace preconditioner
} // namespace utils

#endif // PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_PRECONDITIONRS_PRECONDITIONER_BASE_HPP_
