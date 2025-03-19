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

#ifndef PYTHON_BINDING_NTREE_HPP
#define PYTHON_BINDING_NTREE_HPP

#include <ttns_lib/ttn/tree/ntree.hpp>
#include <ttns_lib/ttn/tree/ntree_builder.hpp>

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

template <typename T> void init_ntree_node(py::module &m) {
  using namespace ttns;
  // wrapper for the ntree_node type
  using node_type = ntree_node<ntree<T>>;
  py::class_<node_type>(m, "ntreeNode")
      .def(py::init(), "Construct an empty ntree node object.")
      .def("level", &node_type::level, R"mydelim(

            :returns: The level of the node in the tree.  root = 0, child of root = 1, etc.
            :rtype: int

            )mydelim")
      .def("nleaves", &node_type::nleaves, R"mydelim(

            :returns: The number of leaves in the subtree that this node is the root of
            :rtype: int

            )mydelim")
      .def("size", &node_type::size, R"mydelim(

            :returns: The number of children of this node
            :rtype: int

            )mydelim")
      .def("subtree_size", &node_type::subtree_size, R"mydelim(

            :returns: The number of nodes in the subtree that this node is the root of
            :rtype: int

            )mydelim")
      .def("empty", &node_type::empty, R"mydelim(

            :returns: Whether or not the node has any children.
            :rtype: bool

            )mydelim")
      .def("is_root", &node_type::is_root, R"mydelim(

            :returns: Whether or not the node is the root node.
            :rtype: bool

            )mydelim")
      .def("is_leaf", &node_type::is_leaf, R"mydelim(

            :returns: Whether or not the node is a leaf node.
            :rtype: bool

            )mydelim")
      .def("leaf_indices", &node_type::leaf_indices, py::arg(),
           py::arg("resize") = true)
      .def(
          "leaf_indices",
          [](const node_type &n, bool resize = true) {
            std::vector<std::vector<size_t>> linds;
            n.leaf_indices(linds, resize);
            return linds;
          },
          py::arg("resize") = true)
      .def("index", &node_type::index)
      .def("index",
           [](const node_type &n) {
             std::vector<size_t> linds;
             n.index(linds);
             return linds;
           })

      .def("parent", &node_type::parent)
      .def("clear", &node_type::clear)
      .def_property(
          "value",
          static_cast<const T &(node_type::*)() const>(&node_type::value),
          [](node_type &o, const T &i) { o.value() = i; })
      .def_property(
          "data",
          static_cast<const T &(node_type::*)() const>(&node_type::data),
          [](node_type &o, const T &i) { o.data() = i; })
      .def("at",
           static_cast<const node_type &(node_type::*)(size_t) const>(
               &node_type::at),
           py::return_value_policy::reference)
      .def("at", static_cast<node_type &(node_type::*)(size_t)>(&node_type::at),
           py::return_value_policy::reference)
      .def("at",
           static_cast<const node_type &(
               node_type::*)(const std::vector<size_t> &, size_t) const>(
               &node_type::at),
           py::return_value_policy::reference)
      .def("at",
           static_cast<node_type &(node_type::*)(const std::vector<size_t> &,
                                                 size_t)>(&node_type::at),
           py::return_value_policy::reference)
      .def("__getitem__",
           static_cast<node_type &(node_type::*)(size_t)>(
               &node_type::operator[]),
           py::return_value_policy::reference)
      .def("back",
           static_cast<const node_type &(node_type::*)() const>(
               &node_type::back),
           py::return_value_policy::reference)
      .def("back", static_cast<node_type &(node_type::*)()>(&node_type::back),
           py::return_value_policy::reference)
      .def("front",
           static_cast<const node_type &(node_type::*)() const>(
               &node_type::front),
           py::return_value_policy::reference)
      .def("front", static_cast<node_type &(node_type::*)()>(&node_type::front),
           py::return_value_policy::reference)
      .def("insert", static_cast<size_t (node_type::*)(const node_type &)>(
                         &node_type::insert))
      .def("insert",
           static_cast<size_t (node_type::*)(const T &)>(&node_type::insert))
      .def("insert_front",
           static_cast<size_t (node_type::*)(const node_type &)>(
               &node_type::insert_front))
      .def("insert_front", static_cast<size_t (node_type::*)(const T &)>(
                               &node_type::insert_front))
      .def("remove",
           static_cast<void (node_type::*)(size_t)>(&node_type::remove));
}

template <typename T> void init_ntree(py::module &m) {
  using namespace ttns;
  using node_type = ntree_node<ntree<T>>;
  // wrapper for the ntree_node type
  py::class_<ntree<T>>(m, "ntree")
      .def(py::init(), "Default construct an empty ntree.")
      .def(py::init<const ntree<T> &>(), R"mydelim(
            Copy construct an ntree from another ntree object

            :Parameters:    - **tree** (:class:`ntree`) - Input ntree to construct a copy of

            )mydelim")
      .def(py::init<const std::string &>(), R"mydelim(
            Copy construct an ntree from a string representation of a tree.  Here we use a set of nested parentheses to represent
            a tree e.g.:

            (1(2(3)(4))(5(6)(7)) 

            corresponds to the tree

                  1
                 x x
                x   x
               2     5
              x x   x x  
             x   x x   x
            3    4 6    7

            :Parameters:    - **tree** (str) - A string representation of a tree

            )mydelim")

      .def("__copy__", [](const ntree<T> &o) { return ntree<T>(o); })
      .def(
          "__deepcopy__",
          [](const ntree<T> &o, py::dict) { return ntree<T>(o); },
          py::arg("memo"))

      .def("assign", &ntree<T>::operator=, R"mydelim(
            Assign the value of this ntree using another ntree

            :param tree: Input ntree to construct a copy of
            :type tree: ntree

            )mydelim")

      .def("empty", &ntree<T>::empty, R"mydelim(

            :returns: Returns whether or not the ntree object is empty.  That is it contains no nodes.
            :rtype: bool

            )mydelim")
      .def("nleaves", &ntree<T>::nleaves, R"mydelim(

            :returns: The number of leaf nodes in the ntree
            :rtype: int

            )mydelim")
      .def("size", &ntree<T>::size, R"mydelim(

            :returns: The total number of nodes in the ntree
            :rtype: int

            )mydelim")
      .def("load", &ntree<T>::load, R"mydelim(
            Load an ntree object from a string representation of a tree.  Here we use a set of nested parentheses to represent
            a tree e.g.:

            (1(2(3)(4))(5(6)(7)) 

            corresponds to the tree

                  1
                 x x
                x   x
               2     5
              x x   x x  
             x   x x   x
            3    4 6    7

            :Parameters:    - **tree** (str) - A string representation of a tree

            )mydelim")
      .def("insert", &ntree<T>::insert, R"mydelim(
            Insert a new root node into the tree.  If the tree already contains a root node this will throw an exception. 

            :param val: The data to store at the root node
            :type val: int
            )mydelim")
      .def("insert_at", &ntree<T>::insert_at, R"mydelim(
            Insert an element as a child of a specific node in the tree.  Here the position is defined by a list of numbers that define which child to traverse down at each node of the tree. 
            If the list is empty this attempts to insert a new root.

            :param inds: The index of the node where we aim to insert a child.  
            :type inds: list[int]
            :param val: The data to store at the root node
            :type val: int

            )mydelim")
      .def("leaf_indices", &ntree<T>::leaf_indices, R"mydelim(
            Set the value of an input list to contain all of the leaf indices of the tree. 

            :param inds: The indices of each of the leaf nodes of the tree
            :type inds: list[list[int]]

            )mydelim")
      .def(
          "leaf_indices",
          [](const ntree<T> &n) {
            std::vector<std::vector<size_t>> linds;
            n.leaf_indices(linds);
            return linds;
          },
          R"mydelim(

            :returns: The indices of each of the leaf nodes of the tree
            :rtype: list[list[int]]

            )mydelim")
      .def("clear", &ntree<T>::clear,
           "Deallocate all internal storage of the ntree object.  Setting it "
           "so that it has no nodes.")
      .def("__len__", &ntree<T>::size, R"mydelim(

            :returns: The total number of nodes in the ntree
            :rtype: int

            )mydelim")
      .def(
          "__iter__",
          [](ntree<T> &s) { return py::make_iterator(s.begin(), s.end()); },
          py::keep_alive<0, 1>(),
          R"mydelim(

                :returns: An iterator for traversing the list in depth first search order

                )mydelim")
      .def(
          "dfs",
          [](ntree<T> &s) {
            return py::make_iterator(s.dfs_begin(), s.dfs_end());
          },
          py::keep_alive<0, 1>(),
          R"mydelim(

                :returns: An iterator for traversing the list in depth first search order

                )mydelim")
      .def(
          "post_order_dfs",
          [](ntree<T> &s) {
            return py::make_iterator(s.post_begin(), s.post_end());
          },
          py::keep_alive<0, 1>(),
          R"mydelim(

                :returns: An iterator for traversing the list in post oder depth first search order

                )mydelim")
      .def(
          "euler_tour",
          [](ntree<T> &s) {
            return py::make_iterator(s.euler_begin(), s.euler_end());
          },
          py::keep_alive<0, 1>(),
          R"mydelim(

                :returns: An iterator for performing an euler tour of the nodes 

                )mydelim")
      .def(
          "bfs",
          [](ntree<T> &s) {
            return py::make_iterator(s.bfs_begin(), s.bfs_end());
          },
          py::keep_alive<0, 1>(),
          R"mydelim(

                :returns: An iterator for traversing the list in breadth first search order
                
                )mydelim")
      .def(
          "leaves",
          [](ntree<T> &s) {
            return py::make_iterator(s.leaf_begin(), s.leaf_end());
          },
          py::keep_alive<0, 1>(),
          R"mydelim(

                :returns: An iterator for traversing the leaves of the tree from left to right

                )mydelim")
      .def("__call__",
           static_cast<const node_type &(ntree<T>::*)() const>(
               &ntree<T>::operator()),
           py::return_value_policy::reference,
           R"mydelim(

                :Returns: The value of the root node
                :Return Type: int

                )mydelim")
      .def("__call__",
           static_cast<node_type &(ntree<T>::*)()>(&ntree<T>::operator()),
           py::return_value_policy::reference,
           R"mydelim(

                :Returns: The value of the root node
                :Return Type: int

                )mydelim")
      .def("__getitem__",
           static_cast<node_type &(ntree<T>::*)(size_t)>(&ntree<T>::operator[]),
           py::return_value_policy::reference,
           R"mydelim(
                Returns the value stored at the ith child of the root

                :param i: The index of the node we aim to return
                :type i: int

                :returns: The value of the root node
                :rtype: int

                )mydelim")
      .def("at",
           static_cast<const node_type &(
               ntree<T>::*)(const std::vector<size_t> &) const>(&ntree<T>::at),
           py::return_value_policy::reference, R"mydelim(
                Returns a constant reference to a specific node in the tree

                :Paramters:     - **inds** (list[int]) - The index of the node where we aim to insert a child.  

                :Returns: A constant reference to the specified node
                :Return Type: :class:`ntreeNode`

                )mydelim")
      .def("at",
           static_cast<node_type &(ntree<T>::*)(const std::vector<size_t> &)>(
               &ntree<T>::at),
           py::return_value_policy::reference, R"mydelim(
                Returns a reference to a specific node in the tree

                :Paramters:     - **inds** (list[int]) - The index of the node where we aim to insert a child.  

                :Returns: A reference to the specified node
                :Return Type: :class:`ntreeNode`

                )mydelim")
      .def("root",
           static_cast<const node_type &(ntree<T>::*)() const>(&ntree<T>::root),
           py::return_value_policy::reference,
           R"mydelim(

                :Returns: A constant reference to the root node
                :Return Type: :class:`ntreeNode`

                )mydelim")
      .def("root", static_cast<node_type &(ntree<T>::*)()>(&ntree<T>::root),
           py::return_value_policy::reference,
           R"mydelim(

                :Returns: A reference to the root node
                :Return Type: :class:`ntreeNode`

                )mydelim")
      .def(
          "__str__",
          [](const ntree<T> &o) {
            std::stringstream oss;
            oss << o;
            return oss.str();
          },
          R"mydelim(

                :returns: A string representation of the tree using the nested parentheses format.
                :rtype: str

                )mydelim"

          )
      .def(
          "as_json",
          [](const ntree<T> &o) {
            std::stringstream oss;
            o.as_json(oss);
            return oss.str();
          },
          R"mydelim(

                :returns: A json string representing the tree
                :rtype: str

                )mydelim")
      .doc() = R"mydelim(
        A class for handling a generic tree structure that stores an integer at each node.  Within pyTTN this class
        is used to represent the topology of the tree tensor network, with the integers stored at a node corresponding
        to the number of single-particle functions (basis functions) associated with this node.  Alternatively, this
        can be viewed as the bond-dimension connecting the node with its parent.  In the current version we require the root node
        to have a value of 1, which corresponds to a topology representing a single state. 
        )mydelim";
}

template <typename T> void init_ntree_builder(py::module &m) {
  using namespace ttns;
  // wrap the ntree_builder c++ class
  py::class_<ntree_builder<T>>(m, "ntreeBuilder")
      // wrap the sanitise function for taking an ntree and ensuring it can is a
      // valid topology for a hierarchical tucker object
      .def_static("sanitise", &ntree_builder<T>::sanitise_tree, py::arg(),
                  py::arg("remove_bond_matrices") = true, R"mydelim(
                Sanitise an ntree to ensure that it corresponds to a valid structure for TTN and if specified does not have any bond matrices.

                :param tree: The tree to sanitise.  This tree is overwritten by the sanitised structure.
                :type tree: ntree
                :param remove_bond_matrices: A flag as to whether or not to remove bond matrices.  (Default: True)
                :type tree: bool, optional

                )mydelim")
      .def_static("insert_basis_nodes", &ntree_builder<T>::insert_basis_nodes,
                  R"mydelim(
                Insert local basis nodes within the ntree object to ensure the structure is in the format expected by the TTN class.

                :param tree: The tree to edit.  This tree is overwritten in this process.
                :type tree: ntree

                )mydelim")
      .def_static("sanitise_bond_dimensions",
                  &ntree_builder<T>::sanitise_bond_dimensions, R"mydelim(
                Alter the bond dimensions within the tree such that we do not have any bonds where we have a larger dimension than the number of states.

                :param tree: The tree to edit.  This tree is overwritten in this process.
                :type tree: ntree

                )mydelim")
      .def_static("collapse_bond_matrices",
                  &ntree_builder<T>::collapse_bond_matrices, R"mydelim(
                Edit the tree collapsing all bond matrices into adjacent tensors.

                :param tree: The tree to edit.  This tree is overwritten in this process.
                :type tree: ntree

                )mydelim")

      // construct balanced ml-mctdh trees and subtrees
      .def_static(
          "mlmctdh_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &, size_t, size_t &&,
                                   bool)>(ntree_builder<T>::htucker_tree),
          py::arg(), py::arg(), py::arg(),
          py::arg("include_local_basis_transformation") = true,
          R"mydelim(
                        Construct a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using an integer to specify the internal bond dimensions.

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **include_local_basis_transformation** (bool, optional) - A flag as to or not to include a set of bond matrices at the leaves of the tree defining a local basis transformation.  (Default: True)
                        
                        :Returns: A balanced tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")
      .def_static(
          "mlmctdh_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &, size_t,
                                   const std::function<size_t(size_t)> &,
                                   bool)>(ntree_builder<T>::htucker_tree),
          py::arg(), py::arg(), py::arg(),
          py::arg("include_local_basis_transformation") = true,
          R"mydelim(
                        Construct a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **include_local_basis_transformation** (bool, optional) - A flag as to or not to include a set of bond matrices at the leaves of the tree defining a local basis transformation.  (Default: True)
                        
                        :Returns: A balanced tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")
      .def_static(
          "mlmctdh_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &, size_t, size_t &&,
                                   T)>(ntree_builder<T>::htucker_tree),
          R"mydelim(
                        Construct a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (int) - The variable used for defining the number of individual mode states to retain in the tree structure
                        
                        :Returns: A balanced tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`
                        )mydelim")
      .def_static(
          "mlmctdh_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &, size_t,
                                   const std::function<size_t(size_t)> &, T)>(
              ntree_builder<T>::htucker_tree),
          R"mydelim(
                        Construct a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **lhd** (int) - The variable used for defining the number of individual mode states to retain in the tree structure
                        
                        :Returns: A balanced tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`
                        )mydelim")
      .def_static("mlmctdh_tree",
                  static_cast<ntree<T> (*)(const std::vector<T> &, size_t,
                                           size_t &&, const std::vector<T> &)>(
                      ntree_builder<T>::htucker_tree),
                  R"mydelim(
                        Construct a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (list[int]) - The variable used for defining the number of individual mode states to retain in the tree structure.  Here this is different for each leaf of the tree.
                        
                        :Returns: A balanced tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`
                        )mydelim")
      .def_static(
          "mlmctdh_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &, size_t,
                                   const std::function<size_t(size_t)> &,
                                   const std::vector<T> &)>(
              ntree_builder<T>::htucker_tree),
          R"mydelim(
                        Construct a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **lhd** (list[int]) - The variable used for defining the number of individual mode states to retain in the tree structure.  Here this is different for each leaf of the tree.
                        
                        :Returns: A balanced tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`
                        )mydelim")

      .def_static(
          "mlmctdh_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t, const std::function<size_t(size_t)> &,
                               bool)>(ntree_builder<T>::htucker_subtree),
          py::arg(), py::arg(), py::arg(), py::arg(),
          py::arg("include_local_basis_transformation") = true,
          R"mydelim(
                        Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using an integer to specify the internal bond dimensions.

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **include_local_basis_transformation** (bool, optional) - A flag as to or not to include a set of bond matrices at the leaves of the tree defining a local basis transformation.  (Default: True)
                        )mydelim")
      .def_static(
          "mlmctdh_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t, size_t &&, bool)>(
              ntree_builder<T>::htucker_subtree),
          py::arg(), py::arg(), py::arg(), py::arg(),
          py::arg("include_local_basis_transformation") = true,
          R"mydelim(
                        Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **include_local_basis_transformation** (bool, optional) - A flag as to or not to include a set of bond matrices at the leaves of the tree defining a local basis transformation.  (Default: True)
                        )mydelim")
      .def_static(
          "mlmctdh_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t, const std::function<size_t(size_t)> &,
                               T)>(ntree_builder<T>::htucker_subtree),
          R"mydelim(
                        Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (int) - The variable used for defining the number of individual mode states to retain in the tree structure
                        
                        )mydelim")
      .def_static(
          "mlmctdh_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t, size_t &&, T)>(
              ntree_builder<T>::htucker_subtree),
          R"mydelim(
                        Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **lhd** (int) - The variable used for defining the number of individual mode states to retain in the tree structure
                        
                        )mydelim")
      .def_static(
          "mlmctdh_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t, const std::function<size_t(size_t)> &,
                               const std::vector<T> &)>(
              ntree_builder<T>::htucker_subtree),
          R"mydelim(
                        Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (list[int]) - The variable used for defining the number of individual mode states to retain in the tree structure.  Here this is different for each leaf of the tree.
                        
                        )mydelim")
      .def_static(
          "mlmctdh_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t, size_t &&, const std::vector<T> &)>(
              ntree_builder<T>::htucker_subtree),
          R"mydelim(
                        Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d, using a function that depends on the distance from the root to specify the internal bond dimensions.
                        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **degree** (int) - The degree of the tree structure.  2 corresponds to a balanced binary tree.
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **lhd** (list[int]) - The variable used for defining the number of individual mode states to retain in the tree structure.  Here this is different for each leaf of the tree.
                        
                        )mydelim")
      // construct degenerate trees representing mps's
      .def_static("mps_tree",
                  static_cast<ntree<T> (*)(const std::vector<T> &, size_t &&)>(
                      ntree_builder<T>::mps_tree),
                  R"mydelim(
                        Construct a new mps ntree for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimensions.

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                        
                        :Returns: A MPS tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")
      .def_static(
          "mps_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &, size_t &&,
                                   size_t &&)>(ntree_builder<T>::mps_tree),
          R"mydelim(
                        Construct a new mps ntree for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimension. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by an integer. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (int) - A vector specifying the local basis size.
                        
                        :Returns: A MPS tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")
      .def_static(
          "mps_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &,
                                   const std::function<size_t(size_t)> &)>(
              ntree_builder<T>::mps_tree),
          R"mydelim(
                        Construct a new mps ntree for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimensions.

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                        
                        :Returns: A MPS tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")
      .def_static(
          "mps_tree",
          static_cast<ntree<T> (*)(const std::vector<T> &,
                                   const std::function<size_t(size_t)> &,
                                   const std::function<size_t(size_t)> &)>(
              ntree_builder<T>::mps_tree),
          R"mydelim(
                        Construct a new mps ntree for representing a tensor with dimensions specified in a list using a function that depends on the distance from the root to specify the internal bond dimensions. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by a second function. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **flhd** (callable(int)) - A function specifying the local basis size.
                        
                        :Returns: A MPS tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")

      .def_static("mps_tree",
                  static_cast<ntree<T> (*)(const std::vector<T> &, size_t &&,
                                           const std::vector<size_t> &)>(
                      ntree_builder<T>::mps_tree),
                  R"mydelim(
                        Construct a new mps ntree for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimension. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by an array. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (list[int]) - A vector specifying the local basis size.
                        
                        :Returns: A MPS tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")
      .def_static(
          "mps_tree",
          static_cast<ntree<T> (*)(
              const std::vector<T> &, const std::function<size_t(size_t)> &,
              const std::vector<size_t> &)>(ntree_builder<T>::mps_tree),
          R"mydelim(
                        Construct a new mps ntree for representing a tensor with dimensions specified in a list using a function that depends on the distance from the root to specify the internal bond dimensions. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by an array. 

                        :Parameters:    - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **lhd** (list[int]) - A vector specifying the local basis size.
                        
                        :Returns: A MPS tree with the structure determined by the users inputs
                        :Return Type: :class:`ntree`

                        )mydelim")
      .def_static(
          "mps_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t &&)>(ntree_builder<T>::mps_subtree),
          R"mydelim(
                        Append a new mps to a node in an ntree for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimensions.

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                        )mydelim")
      .def_static(
          "mps_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t &&, size_t &&)>(
              ntree_builder<T>::mps_subtree),
          R"mydelim(
                        Append a new mps to a node in an ntree  for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimension. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by an integer. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (int) - A vector specifying the local basis size.
                        
                        )mydelim")
      .def_static(
          "mps_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               const std::function<size_t(size_t)> &)>(
              ntree_builder<T>::mps_subtree),
          R"mydelim(
                        Append a new mps to a node in an ntree for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimensions.

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  

                        )mydelim")
      .def_static(
          "mps_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               const std::function<size_t(size_t)> &,
                               const std::function<size_t(size_t)> &)>(
              ntree_builder<T>::mps_subtree),
          R"mydelim(
                        Append a new mps to a node in an ntree for representing a tensor with dimensions specified in a list using a function that depends on the distance from the root to specify the internal bond dimensions. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by a second function. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **flhd** (callable(int)) - A function specifying the local basis size.

                        )mydelim")
      .def_static(
          "mps_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               size_t &&, const std::vector<size_t> &)>(
              ntree_builder<T>::mps_subtree),
          R"mydelim(
                        Append a new mps to a node in an ntree for representing a tensor with dimensions specified in a list using an integer to specify the internal bond dimension. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by an array. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **chi** (int) - A fixed internal bond dimension to use throughout the tree.
                                        - **lhd** (list[int]) - A vector specifying the local basis size.

                        )mydelim")
      .def_static(
          "mps_subtree",
          static_cast<void (*)(ntree_node<ntree<T>> &, const std::vector<T> &,
                               const std::function<size_t(size_t)> &,
                               const std::vector<size_t> &)>(
              ntree_builder<T>::mps_subtree),
          R"mydelim(
                        Append a new mps to a node in an ntree for representing a tensor with dimensions specified in a list using a function that depends on the distance from the root to specify the internal bond dimensions. 
                        Here the tree also automatically includes local optimised bases with the number of states being set by an array. 

                        :Parameters:    - **node** (:class:`ntreeNode`) - The node to append the tree to
                                        - **dims** (list[int]) - The local Hilbert space dimensions of each leaf of the tree
                                        - **fchi** (callable(int)) - A function specifying the bond dimension as a function of depth.  
                                        - **lhd** (list[int]) - A vector specifying the local basis size.
                        
                        )mydelim");
}

void initialise_ntree(py::module &m);

#endif // PYTHON_BINDING_NTREE_HPP
