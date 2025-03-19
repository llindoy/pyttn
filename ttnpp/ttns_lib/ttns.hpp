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

#ifndef PYTTN_TTNS_LIB_TTNS_HPP_
#define PYTTN_TTNS_LIB_TTNS_HPP_

#include <common/tmp_funcs.hpp>

// include the underlying ntree objects that enable efficient tree building
#include "ttn/tree/ntree_builder.hpp"

// the generic hierarchical tucker tensor/tree tensor network object
#include "ttn/ms_ttn.hpp"
#include "ttn/ttn.hpp"

// generic observable
#include "observables/matrix_element.hpp"

#include "op.hpp"

// important operators for acting on TTNs
#include "operators/product_operator.hpp"
#include "operators/site_operators/matrix_operators.hpp"
#include "operators/site_operators/site_operator.hpp"
#include "operators/site_operators/site_product_operator.hpp"

// actual integrator
// #include "algorithms/one_site_gso_engine.hpp"
// #include "algorithms/adaptive_one_site_gso_engine.hpp"
// #include "algorithms/tdvp_integrator.hpp"

#endif // PYTTN_TTNS_LIB_TTNS_HPP_//
