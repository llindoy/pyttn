#ifndef PYTHON_BINDING_NTREE_HPP
#define PYTHON_BINDING_NTREE_HPP

#include <ttns_lib/ttn/tree/ntree.hpp>
#include <ttns_lib/ttn/tree/ntree_builder.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T>
void init_ntree_node(py::module &m)
{
    using namespace ttns;
    //wrapper for the ntree_node type 
    using node_type = ntree_node<ntree<T>>;
    py::class_<node_type>(m, "ntreeNode")
        .def(py::init())
        .def("level", &node_type::level)
        .def("nleaves", &node_type::nleaves)
        .def("size", &node_type::size)
        .def("subtree_size", &node_type::subtree_size)
        .def("empty", &node_type::empty)
        .def("is_root", &node_type::is_root)
        .def("is_leaf", &node_type::is_leaf)
        .def("leaf_indices", &node_type::leaf_indices, py::arg(), py::arg("resize") = true)
        .def(   
                "leaf_indices", 
                [](const node_type& n, bool resize=true)
                {
                    std::vector<std::vector<size_t>> linds;
                    n.leaf_indices(linds, resize);
                    return linds;
                }, 
                py::arg("resize") = true
            )
        .def("parent", &node_type::parent)
        .def("clear", &node_type::clear)
        .def_property
            (
                "value", 
                static_cast<const T& (node_type::*)() const>(&node_type::value),
                [](node_type& o, const T& i){o.value() = i;}
            )
        .def_property
            (
                "data", 
                static_cast<const T& (node_type::*)() const>(&node_type::data),
                [](node_type& o, const T& i){o.data() = i;}
            )
        .def(
                "at", 
                static_cast<const node_type& (node_type::*)(size_t) const>(&node_type::at), 
                py::return_value_policy::reference
            )
        .def(
                "at", 
                static_cast<node_type& (node_type::*)(size_t)>(&node_type::at), 
                py::return_value_policy::reference
            )
        .def(   
                "at", 
                static_cast<const node_type& (node_type::*)(const std::vector<size_t>&, size_t) const>(&node_type::at), 
                py::return_value_policy::reference
            )
        .def(
                "at", 
                static_cast<node_type& (node_type::*)(const std::vector<size_t>&, size_t)>(&node_type::at), 
                py::return_value_policy::reference
            )
        .def(
                "__getitem__", 
                static_cast<const node_type& (node_type::*)(size_t) const>(&node_type::operator[]),
                py::return_value_policy::reference
            )
        .def(
                "back", 
                static_cast<const node_type& (node_type::*)() const>(&node_type::back), 
                py::return_value_policy::reference
            )
        .def(
                "back", 
                static_cast<node_type& (node_type::*)()>(&node_type::back), 
                py::return_value_policy::reference
            )
        .def(
                "front", 
                static_cast<const node_type& (node_type::*)() const>(&node_type::front), 
                py::return_value_policy::reference
            )
        .def(
                "front", 
                static_cast<node_type& (node_type::*)()>(&node_type::front), 
                py::return_value_policy::reference
            )
        .def(
                "insert", 
                static_cast<size_t (node_type::*)(const node_type&)>(&node_type::insert)
            )
        .def(
                "insert", 
                static_cast<size_t (node_type::*)(const T&)>(&node_type::insert)
            )
        .def(
                "remove", 
                static_cast<void (node_type::*)(size_t)>(&node_type::remove)
            );

};

template <typename T>
void init_ntree(py::module &m)
{
    using namespace ttns;
    using node_type = ntree_node<ntree<T>>;
    //wrapper for the ntree_node type 
    py::class_<ntree<T>>(m, "ntree")
        .def(py::init())
        .def(py::init<const ntree<T>&>())
        .def(py::init<const std::string&>())
        .def("assign", &ntree<T>::operator=)    
        .def("empty", &ntree<T>::empty)
        .def("nleaves", &ntree<T>::nleaves)
        .def("size", &ntree<T>::size)
        .def("load", &ntree<T>::load)
        .def("insert", &ntree<T>::insert)
        .def("insert_at", &ntree<T>::insert_at)
        .def("leaf_indices", &ntree<T>::leaf_indices)
        .def(   
                "leaf_indices", 
                [](const ntree<T>& n)
                {
                    std::vector<std::vector<size_t>> linds;
                    n.leaf_indices(linds);
                    return linds;
                }
            )
        .def("clear", &ntree<T>::clear)
        .def("__len__", &ntree<T>::size)
        .def(
                "__iter__",
                [](ntree<T>& s){return py::make_iterator(s.begin(), s.end());},
                py::keep_alive<0, 1>()
            )
        .def(
                "dfs",
                [](ntree<T>& s){return py::make_iterator(s.dfs_begin(), s.dfs_end());},
                py::keep_alive<0, 1>()
            )
        .def(
                "post_order_dfs",
                [](ntree<T>& s){return py::make_iterator(s.post_begin(), s.post_end());},
                py::keep_alive<0, 1>()
            )
        .def(
                "euler_tour",
                [](ntree<T>& s){return py::make_iterator(s.euler_begin(), s.euler_end());},
                py::keep_alive<0, 1>()
            )
        .def(
                "bfs",
                [](ntree<T>& s){return py::make_iterator(s.bfs_begin(), s.bfs_end());},
                py::keep_alive<0, 1>()
            )
        .def(
                "leaves",
                [](ntree<T>& s){return py::make_iterator(s.leaf_begin(), s.leaf_end());},
                py::keep_alive<0, 1>()
            )
        .def(
                "__call__", 
                static_cast<const node_type& (ntree<T>::*)() const>(&ntree<T>::operator()), 
                py::return_value_policy::reference
            )
        .def(      
                "__call__", 
                static_cast<node_type& (ntree<T>::*)()>(&ntree<T>::operator()), 
                py::return_value_policy::reference
            )
        .def(
                "__getitem__", 
                static_cast<const node_type& (ntree<T>::*)(size_t) const>(&ntree<T>::operator[])
            )
        .def(
                "at", 
                static_cast<const node_type& (ntree<T>::*)(const std::vector<size_t>&) const>(&ntree<T>::at), 
                py::return_value_policy::reference
            )
        .def(
                "at", 
                static_cast<node_type& (ntree<T>::*)(const std::vector<size_t>&)>(&ntree<T>::at), 
                py::return_value_policy::reference
            )
        .def(
                "root", 
                static_cast<const node_type& (ntree<T>::*)() const>(&ntree<T>::root), 
                py::return_value_policy::reference
            )
        .def(
                "root", 
                static_cast<node_type& (ntree<T>::*)()>(&ntree<T>::root), 
                py::return_value_policy::reference
            )
        .def(
                "__str__", 
                [](const ntree<T>& o){std::stringstream oss;   oss << o; return oss.str();}
            )
        .def(
                "as_json", 
                [](const ntree<T>& o){std::stringstream oss;   o.as_json(oss); return oss.str();}
            );

}

template <typename T> 
void init_ntree_builder(py::module &m)
{
    using namespace ttns;
    //wrap the ntree_builder c++ class
    py::class_<ntree_builder<T>>(m, "ntreeBuilder")
        //wrap the sanitise function for taking an ntree and ensuring it can is a valid topology for a hierarchical tucker object
        .def_static("sanitise", &ntree_builder<T>::sanitise_tree, py::arg(), py::arg("remove_bond_matrices")=true)
        .def_static("insert_basis_nodes", &ntree_builder<T>::insert_basis_nodes)
        .def_static("sanitise_bond_dimensions", &ntree_builder<T>::sanitise_bond_dimensions)
        .def_static("collapse_bond_matrices", &ntree_builder<T>::collapse_bond_matrices)

        //construct balanced ml-mctdh trees and subtrees
        .def_static(    
                        "mlmctdh_tree", 
                        static_cast<ntree<T>(*)(const std::vector<T>&, size_t, size_t&&)>(ntree_builder<T>::htucker_tree)
                   )
        .def_static(
                        "mlmctdh_tree", 
                        static_cast<ntree<T>(*)(const std::vector<T>&, size_t, const std::function<size_t(size_t)>&)>(ntree_builder<T>::htucker_tree)
                   )
        .def_static(    
                        "mlmctdh_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, size_t, size_t&&)>(ntree_builder<T>::htucker_subtree)
                   )
        .def_static(
                        "mlmctdh_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, size_t, const std::function<size_t(size_t)>&)>(ntree_builder<T>::htucker_subtree)
                   )
        .def_static(    
                        "htucker_tree", 
                        static_cast<ntree<T>(*)(const std::vector<T>&, size_t, size_t&&)>(ntree_builder<T>::htucker_tree)
                   )
        .def_static(
                        "htucker_tree", 
                        static_cast<ntree<T>(*)(const std::vector<T>&, size_t, const std::function<size_t(size_t)>&)>(ntree_builder<T>::htucker_tree)
                   )
        .def_static(    
                        "htucker_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, size_t, size_t&&)>(ntree_builder<T>::htucker_subtree)
                   )
        .def_static(
                        "htucker_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, size_t, const std::function<size_t(size_t)>&)>(ntree_builder<T>::htucker_subtree)
                   )
        //construct degenerate trees representing mps's 
        .def_static(
                        "mps_tree", 
                        static_cast<ntree<T>(*)(const std::vector<T>&, size_t&&)>(ntree_builder<T>::mps_tree)
                   )
        .def_static(
                        "mps_tree", 
                        static_cast<ntree<T>(*)(const std::vector<T>&, const std::function<size_t(size_t)>&)>(ntree_builder<T>::mps_tree)
                   )
        .def_static(
                        "mps_tree", 
                        static_cast<ntree<T>(*)(const std::vector<T>&, const std::function<size_t(size_t)>&, const std::function<size_t(size_t)>&)>(ntree_builder<T>::mps_tree)
                   )
        .def_static(
                        "mps_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, size_t&&)>(ntree_builder<T>::mps_subtree)
                   )
        .def_static(
                        "mps_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, size_t&&, size_t&&)>(ntree_builder<T>::mps_subtree)
                   )
        .def_static(
                        "mps_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, const std::function<size_t(size_t)>&)>(ntree_builder<T>::mps_subtree)
                   )
        .def_static(
                        "mps_subtree", 
                        static_cast<void(*)(ntree_node<ntree<T>>&, const std::vector<T>&, const std::function<size_t(size_t)>&, const std::function<size_t(size_t)>&)>(ntree_builder<T>::mps_subtree)
                   );
}

void initialise_ntree(py::module& m);

#endif //PYTHON_BINDING_NTREE_HPP


