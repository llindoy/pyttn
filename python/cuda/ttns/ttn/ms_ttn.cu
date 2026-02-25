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

#include <ttns_lib/ttn/ms_ttn.hpp>
#include <ttns_lib/ttn/multiset_ttn_slice.hpp>
#include <ttns_lib/ttn/ttn_nodes/ms_ttn_node.hpp>
#include <ttns_lib/operators/site_operators/site_operator.hpp>
#include "../../../pyttn_typedef.hpp"

namespace ttns
{
#ifdef BUILD_REAL_TTN
    template class ms_ttn<pyttn_real_type, linalg::cuda_backend>;
    template class ms_ttn_node<pyttn_real_type, linalg::cuda_backend>;
    template class tree_node<tree_base<std::vector<ttn_node_data<pyttn_real_type, linalg::cuda_backend>>>>;
    template class ttn_node_data<pyttn_real_type, linalg::cuda_backend>;
    template class multiset_ttn_slice<pyttn_real_type, linalg::cuda_backend, false>;
    template class multiset_ttn_slice<pyttn_real_type, linalg::cuda_backend, true>;
#endif
    template class ms_ttn<std::complex<pyttn_real_type>, linalg::cuda_backend>;
    template class tree_node<tree_base<std::vector<ttn_node_data<std::complex<pyttn_real_type>, linalg::cuda_backend>>>>;
    template class ttn_node_data<std::complex<pyttn_real_type>, linalg::cuda_backend>;
    template class multiset_ttn_slice<std::complex<pyttn_real_type>, linalg::cuda_backend, false>;
    template class multiset_ttn_slice<std::complex<pyttn_real_type>, linalg::cuda_backend, true>;
}