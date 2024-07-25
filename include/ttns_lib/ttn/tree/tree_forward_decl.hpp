#ifndef TTNS_TREE_FORWARD_DECL_HPP
#define TTNS_TREE_FORWARD_DECL_HPP

#include <common/exception_handling.hpp>

namespace ttns
{

class tree_node_tag{};
class tree_tag{};

//metaprogram tag to determine if object is tree type
template <typename T>  using is_tree = std::is_base_of<tree_tag, T>;

//metaprogram tag to determine if object is tree type
template <typename T>  using is_tree_node = std::is_base_of<tree_node_tag, T>;

template <typename T>class tree_base;
template <typename T> class tree;
template <typename Tree> class tree_node_base;
template <typename Tree> class tree_node;

template <template <typename, typename> class node_type, typename T, typename backend> class ttn_base;
template <typename T, typename backend> class ttn;
template <typename T, typename backend> class ms_ttn;

}   //namespace ttns
#endif  //  TTNS_TREE_FORWARD_DECL_HPP    //
