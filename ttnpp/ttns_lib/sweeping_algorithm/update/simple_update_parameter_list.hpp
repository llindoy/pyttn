/**                                                                         \
 * This files is part of the pyTTN package.                                 \
 * (C) Copyright 2025 NPL Management Limited                                \
 * Licensed under the Apache License, Version 2.0 (the "License");          \
 * you may not use this file except in compliance with the License.         \
 * You may obtain a copy of the License at                                  \
 *     http://www.apache.org/licenses/LICENSE-2.0                           \
 * Unless required by applicable law or agreed to in writing, software      \
 * distributed under the License is distributed on an "AS IS" BASIS,        \
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. \
 * See the License for the specific language governing permissions and      \
 * limitations under the License                                            \
 */

#ifndef PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_SIMPLE_UPDATE_PARAMETER_LIST_HPP_
#define PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_SIMPLE_UPDATE_PARAMETER_LIST_HPP_

namespace ttns
{
    struct simple_update_parameter_list
    {
        simple_update_parameter_list() : krylov_dim(4), nstep(1) {}
        simple_update_parameter_list(size_t kdim) : krylov_dim(kdim), nstep(1) {}
        simple_update_parameter_list(size_t kdim, size_t ns) : krylov_dim(kdim), nstep(ns) {}
        simple_update_parameter_list(const simple_update_parameter_list &o) = default;
        simple_update_parameter_list(simple_update_parameter_list &&o) = default;
        simple_update_parameter_list &operator=(const simple_update_parameter_list &o) = default;
        simple_update_parameter_list &operator=(simple_update_parameter_list &&o) = default;

        size_t krylov_dim;
        size_t nstep;
    };
}

#endif // PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_SIMPLE_UPDATE_PARAMETER_LIST_HPP_
