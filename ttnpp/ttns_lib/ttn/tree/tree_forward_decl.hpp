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

#ifndef PYTTN_TTNS_LIB_TTN_TREE_TREE_FORWARD_DECL_HPP_
#define PYTTN_TTNS_LIB_TTN_TREE_TREE_FORWARD_DECL_HPP_

#include <common/exception_handling.hpp>

namespace ttns
{

    class tree_node_tag
    {
    };
    class tree_tag
    {
    };

    // metaprogram tag to determine if object is tree type
    template <typename T>
    using is_tree = std::is_base_of<tree_tag, T>;

    // metaprogram tag to determine if object is tree type
    template <typename T>
    using is_tree_node = std::is_base_of<tree_node_tag, T>;

    template <typename T>
    class tree_base;
    template <typename T>
    class tree;
    template <typename Tree>
    class tree_node_base;
    template <typename Tree>
    class tree_node;

    template <template <typename, typename> class node_type, typename T, typename backend>
    class ttn_base;
    template <typename T, typename backend>
    class ttn;
    template <typename T, typename backend>
    class ms_ttn;

} // namespace ttns
#endif //  PYTTN_TTNS_LIB_TTN_TREE_TREE_FORWARD_DECL_HPP_ //
