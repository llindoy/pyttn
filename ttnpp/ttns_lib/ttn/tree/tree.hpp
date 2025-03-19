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

#ifndef PYTTN_TTNS_LIB_TTN_TREE_TREE_HPP_
#define PYTTN_TTNS_LIB_TTN_TREE_TREE_HPP_

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>
#include "tree_forward_decl.hpp"
#include "tree_node.hpp"
#include "ntree.hpp"
#include "node_data_traits.hpp"

#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <list>

namespace ttns
{

    // need to clear this up a bit more
    template <typename T>
    class tree_base : public tree_tag
    {
    public:
        // typedefs for the tree base type
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using value_type = T;
        using reference = value_type &;
        using const_reference = const value_type &;
        using tree_type = tree_base<T>;
        using tree_reference = tree_type &;
        using const_tree_reference = const tree_type &;
        using node_type = tree_node<tree_type>;
        using node_reference = node_type &;
        using const_node_reference = const node_type &;

        using node_container_type = std::vector<node_type>;

        using level_index_type = std::vector<std::vector<size_type>>;

    protected:
        // member variables
        node_container_type m_nodes;
        size_type m_nleaves;
        level_index_type m_level_indexing;

        // size constructor
        tree_base(size_type nnodes) : m_nodes(nnodes) {}

    public:
        tree_base() : m_nodes(), m_nleaves(0) {}

        // constructors which attempt to use the default assignment traits object to construct the tree
        template <typename U, typename Alloc>
        tree_base(const ntree<U, Alloc> &tree)
        {
            using traits = node_data_traits::assignment_traits<T, U>;
            static_assert(traits::is_applicable::value, "Failed to construct tree object.  The assignment operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_ntree(tree, traits()), "Failed to construct tree object from ntree object.");
        }

        tree_base(const tree_base &other)
        {
            using traits = node_data_traits::assignment_traits<T, T>;
            static_assert(traits::is_applicable::value, "Failed to construct tree object.  The assignment operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_tree(other, traits()), "Failed to construct tree object from other tree object.");
        }

        template <typename U, typename... Args>
        tree_base(const tree_base<U> &other, Args &&...args)
        {
            using traits = node_data_traits::assignment_traits<T, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to construct tree object.  The assignment operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_tree(other, traits(), std::forward<Args>(args)...), "Failed to construct tree object from other tree object and additional arguments.");
        }

        template <typename U, typename... Args>
        tree_base(const tree_base<U> &other, const tree_base<U> &other2, Args &&...args)
        {
            using traits = node_data_traits::assignment_traits<T, U, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to construct tree object.  The assignment operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_two_trees(other, other2, traits(), std::forward<Args>(args)...), "Failed to construct tree object from other tree object and additional arguments.");
        }

        // constructors which take an alternative assignment functor to construct the nodes
        template <typename U, typename Alloc, typename Func>
        tree_base(const ntree<U, Alloc> &tree, Func &&f)
        {
            CALL_AND_HANDLE(construct_from_ntree(tree, std::forward<Func>(f)), "Failed to construct tree object from ntree object.");
        }

        template <typename Func>
        tree_base(const tree_base &other, Func &&f)
        {
            CALL_AND_HANDLE(construct_from_tree(other, std::forward<Func>(f)), "Failed to construct tree object from other tree object.");
        }

        template <typename U, typename Func, typename... Args>
        tree_base(const tree_base<U> &other, Func &&f, Args &&...args)
        {
            CALL_AND_HANDLE(construct_from_tree(other, std::forward<Func>(f), std::forward<Args>(args)...), "Failed to construct tree object from other tree object and additional arguments.");
        }

        template <typename U, typename Func, typename... Args>
        tree_base(const tree_base<U> &other, const tree_base<U> &other2, Func &&f, Args &&...args)
        {
            CALL_AND_HANDLE(construct_from_two_trees(other, other2, std::forward<Func>(f), std::forward<Args>(args)...), "Failed to construct tree object from other tree object and additional arguments.");
        }

        tree_base(tree_type &&o) = default;
        ~tree_base() {}

        // assignment operator.  This uses the default assignment operator between T and U
        tree_reference operator=(const_tree_reference other)
        {
            using traits = node_data_traits::assignment_traits<T, T>;
            static_assert(traits::is_applicable::value, "Failed to copy assign tree object.  The assignment operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_tree(other, traits()), "Failed to assign tree object from other tree object.");
            return *this;
        }

        template <typename U>
        tree_reference operator=(const tree_base<U> &other)
        {
            using traits = node_data_traits::assignment_traits<T, U>;
            static_assert(traits::is_applicable::value, "Failed to copy assign tree object.  The assignment operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_tree(other, traits()), "Failed to assign tree object from other tree object.");
            return *this;
        }

        template <typename U, typename Alloc>
        tree_reference operator=(const ntree<U, Alloc> &tree)
        {
            using traits = node_data_traits::assignment_traits<T, U>;
            static_assert(traits::is_applicable::value, "Failed to copy assign tree object.  The assignment operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_ntree(tree, traits()), "Failed to assign tree object from an ntree object.");
            return *this;
        }

        template <typename U, typename... Args>
        void resize(const tree_base<U> &other, Args &&...args)
        {
            using traits = node_data_traits::resize_traits<T, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to resize tree object.  The resize operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_tree(other, traits(), std::forward<Args>(args)...), "Failed to resize object.");
        }

        template <typename U, typename... Args>
        void resize(const tree_base<U> &other, const tree_base<U> &other2, Args &&...args)
        {
            using traits = node_data_traits::resize_traits<T, U, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to resize tree object.  The resize operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_two_trees(other, other2, traits(), std::forward<Args>(args)...), "Failed to resize object.");
        }

        template <typename U, typename Alloc, typename... Args>
        void resize(const ntree<U, Alloc> &tree, Args &&...args)
        {
            using traits = node_data_traits::resize_traits<T, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to resize tree object.  The resize operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_ntree(tree, traits(), std::forward<Args>(args)...), "Failed to resize object.");
        }

        template <typename U, typename... Args>
        void reallocate(const tree_base<U> &other, Args &&...args)
        {
            using traits = node_data_traits::reallocate_traits<T, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to reallocate tree object.  The reallocate operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_tree(other, traits(), std::forward<Args>(args)...), "Failed to reallocate object.");
        }

        template <typename U, typename... Args>
        void reallocate(const tree_base<U> &other, const tree_base<U> &other2, Args &&...args)
        {
            using traits = node_data_traits::reallocate_traits<T, U, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to reallocate tree object.  The reallocate operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_two_trees(other, other2, traits(), std::forward<Args>(args)...), "Failed to reallocate object.");
        }

        template <typename U, typename Alloc, typename... Args>
        void reallocate(const ntree<U, Alloc> &tree, Args &&...args)
        {
            using traits = node_data_traits::reallocate_traits<T, U, typename std::decay<Args>::type...>;
            static_assert(traits::is_applicable::value, "Failed to reallocate tree object.  The reallocate operator is not applicable with the input type.");
            CALL_AND_HANDLE(construct_from_ntree(tree, traits(), std::forward<Args>(args)...), "Failed to reallocate object.");
        }

        size_type nleaves() const noexcept { return m_nleaves; }
        size_type size() const noexcept { return m_nodes.size(); }
        bool empty() const noexcept { return m_nodes.size() == 0; }

        const_node_reference root() const
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            return m_nodes[0];
        }
        node_reference root()
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            return m_nodes[0];
        }
        const_node_reference operator[](size_type i) const { return m_nodes[i]; }
        node_reference operator[](size_type i) { return m_nodes[i]; }
        const_node_reference at(size_type i) const
        {
            ASSERT(i < m_nodes.size(), "Failed to access node in tree, element out of bounds.");
            return m_nodes[i];
        }
        node_reference at(size_type i)
        {
            ASSERT(i < m_nodes.size(), "Failed to access node in tree, element out of bounds.");
            return m_nodes[i];
        }

        node_reference operator()(const std::list<size_type> &tree_index)
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            if (tree_index.size() == 0)
            {
                return m_nodes[0];
            }
            else
            {
                return m_nodes[0].at(tree_index.begin(), tree_index.end());
            }
        }
        const_node_reference operator()(const std::list<size_type> &tree_index) const
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            if (tree_index.size() == 0)
            {
                return m_nodes[0];
            }
            else
            {
                return m_nodes[0].at(tree_index.begin(), tree_index.end());
            }
        }
        node_reference at(const std::list<size_type> &tree_index)
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            if (tree_index.size() == 0)
            {
                return m_nodes[0];
            }
            else
            {
                return m_nodes[0].at(tree_index.begin(), tree_index.end());
            }
        }
        const_node_reference at(const std::list<size_type> &tree_index) const
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            if (tree_index.size() == 0)
            {
                return m_nodes[0];
            }
            else
            {
                return m_nodes[0].at(tree_index.begin(), tree_index.end());
            }
        }

        size_type id_at(const std::list<size_type> &tree_index) const
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            if (tree_index.size() == 0)
            {
                return m_nodes[0].id();
            }
            else
            {
                return m_nodes[0].id_at(tree_index.begin(), tree_index.end());
            }
        }

        template <typename IT>
        node_reference operator()(IT first, IT last)
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            if (first == last)
            {
                return m_nodes[0];
            }
            else
            {
                return m_nodes[0].at(++first, last);
            }
        }
        template <typename IT>
        const_node_reference operator()(IT first, IT last) const
        {
            ASSERT(m_nodes.size() != 0, "Failed to access root node of tree. The tree is empty.");
            if (first == last)
            {
                return m_nodes[0];
            }
            else
            {
                return m_nodes[0].at(++first, last);
            }
        }

        void clear()
        {
            for (auto &node : m_nodes)
            {
                CALL_AND_HANDLE(node.clear(), "Unable to clear tree object. Failed when clearing a node.");
            }
            m_nodes.clear();
            m_nleaves = 0;
        }
#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void save(archive &ar) const
        {
            std::vector<serialisation_node_save_wrapper<node_type, size_type>> m_saved_nodes(m_nodes.size());
            for (size_type i = 0; i < m_saved_nodes.size(); ++i)
            {
                m_saved_nodes[i].initialise(&m_nodes[i]);
            }
            CALL_AND_HANDLE(ar(cereal::make_nvp("nodes", m_saved_nodes)), "Failed to serialise tree_base object.  Failed to serialise its nodes.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("nleaves", m_nleaves)), "Failed to serialise tree_base object.  Failed to serialise its number of leaves.");
        }

        template <typename archive>
        void load(archive &ar)
        {
            // create
            std::vector<serialisation_node_load_wrapper<node_type, size_type>> m_loaded_nodes;
            CALL_AND_HANDLE(ar(cereal::make_nvp("nodes", m_loaded_nodes)), "Failed to serialise tree_base object.  Failed to serialise its nodes.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("nleaves", m_nleaves)), "Failed to serialise tree_base object.  Failed to serialise its number of leaves.");

            // now that we have loaded in all of the node information we need to construct
            m_nodes.resize(m_loaded_nodes.size());

            // first pass to set up the data stored in the nodes
            for (size_type i = 0; i < m_loaded_nodes.size(); ++i)
            {
                m_loaded_nodes[i].move_to_node(m_nodes[i]);
            }

            // now that we have the data stored in a correctly packed array we need to set up the correct pointer chain
            for (size_type i = 0; i < m_loaded_nodes.size(); ++i)
            {
                // set up the parent node
                if (m_loaded_nodes[i].is_root)
                {
                    m_nodes[i].m_parent = nullptr;
                }
                else
                {
                    m_nodes[i].m_parent = &m_nodes[m_loaded_nodes[i].m_parent_id];
                }

                // load all of the children nodes.
                m_nodes[i].m_children.resize(m_loaded_nodes[i].m_children_ids.size());
                for (size_type j = 0; j < m_loaded_nodes[i].m_children_ids.size(); ++j)
                {
                    m_nodes[i].m_children[j] = &m_nodes[m_loaded_nodes[i].m_children_ids[j]];
                }
            }
        }
#endif

        template <typename Itype>
        typename std::enable_if<std::is_integral<Itype>::value, void>::type get_edges(std::vector<std::pair<Itype, Itype>> &edges) const
        {
            edges.clear();
            for (size_type ni = 0; ni < m_nodes.size(); ++ni)
            {
                for (size_type mi = 0; mi < m_nodes[ni].size(); ++mi)
                {
                    edges.push_back(std::make_pair(Itype(m_nodes[ni].id()), Itype(m_nodes[ni].child(mi).id())));
                }
            }
        }

    public:
        template <typename U>
        void construct_topology(const tree_base<U> &other)
        {
            using traits = node_data_traits::default_initialisation_traits<T>;
            CALL_AND_HANDLE(construct_from_tree(other, traits()), "Failed to construct topology of tree object.");
        }

        template <typename U, typename Alloc>
        void construct_topology(const ntree<U, Alloc> &other)
        {
            using traits = node_data_traits::default_initialisation_traits<T>;
            CALL_AND_HANDLE(construct_from_ntree(other, traits()), "Failed to construct topology of tree object.");
        }

    protected:
        template <typename U, typename Func, typename... Args>
        void construct_from_tree(const tree_base<U> &other, Func &&f, Args &&...args)
        {
            // make sure that we are not pointing to the same tree object
            if (!same_reference(other))
            {
                const auto &ns = *this;
                if (!has_same_structure(ns, other))
                {
                    CALL_AND_HANDLE(clear(), "Failed to copy construct tree object from a tree object.  Unable to clear the current tree.");
                    if (!other.empty())
                    {
                        CALL_AND_HANDLE(m_nodes.resize(other.size()), "Failed to copy construct tree object from a tree object.  Error when allocating the node container.");

                        m_nleaves = other.nleaves();
                        m_nodes[0].m_parent = nullptr;

                        // we willl probably need to add some additional functions here
                        for (size_type i = 0; i < other.size(); ++i)
                        {
                            CALL_AND_HANDLE(f(m_nodes[i].m_data, other[i](), std::forward<Args>(args)...), "Failed to apply functor to tree nodes");

                            m_nodes[i].m_level = other[i].level();
                            m_nodes[i].m_id = other[i].id();
                            m_nodes[i].m_child_id = other[i].child_id();
                            m_nodes[i].m_index = other[i].index();
                            if (other[i].is_leaf())
                            {
                                m_nodes[i].m_lid = other[i].leaf_index();
                            }
                            else
                            {
                                m_nodes[i].m_lid = m_nleaves;
                            }

                            m_nodes[i].m_children.resize(other[i].size());
                            for (size_type j = 0; j < other[i].size(); ++j)
                            {
                                m_nodes[i].m_children[j] = &m_nodes[other[i].child(j).id()];
                                m_nodes[other[i].child(j).id()].m_parent = &m_nodes[i];
                            }
                        }
                    }
                }
                else
                {
                    for (size_type i = 0; i < other.size(); ++i)
                    {
                        f(m_nodes[i].m_data, other[i](), std::forward<Args>(args)...);
                    }
                }
            }
            initialise_level_indexing();
        }

        template <typename U, typename V, typename Func, typename... Args>
        void construct_from_two_trees(const tree_base<U> &other, const tree_base<V> &other2, Func &&f, Args &&...args)
        {
            ASSERT(has_same_structure(other, other2), "Failed to construct tree object from two tree objects.  The two tree objects are not the same size.");

            // make sure that we are not pointing to the same tree object
            if (!same_reference(other))
            {
                if (!has_same_structure(*this, other))
                {
                    CALL_AND_HANDLE(clear(), "Failed to construct tree object from two tree objects.  Unable to clear the current tree.");
                    if (!other.empty())
                    {
                        CALL_AND_HANDLE(m_nodes.resize(other.size()), "Failed to construct tree object from two tree objects.  Error when allocating the node container.");

                        m_nleaves = other.nleaves();
                        m_nodes[0].m_parent = nullptr;

                        // we willl probably need to add some additional functions here
                        for (size_type i = 0; i < other.size(); ++i)
                        {
                            CALL_AND_HANDLE(f(m_nodes[i].m_data, other[i](), other2[i](), std::forward<Args>(args)...), "Failed to apply functor to tree nodes");

                            m_nodes[i].m_level = other[i].level();
                            m_nodes[i].m_id = other[i].id();
                            m_nodes[i].m_child_id = other[i].child_id();
                            m_nodes[i].m_index = other[i].index();
                            if (other[i].is_leaf())
                            {
                                m_nodes[i].m_lid = other[i].leaf_index();
                            }
                            else
                            {
                                m_nodes[i].m_lid = m_nleaves;
                            }

                            m_nodes[i].m_children.resize(other[i].size());
                            for (size_type j = 0; j < other[i].size(); ++j)
                            {
                                m_nodes[i].m_children[j] = &m_nodes[other[i].child(j).id()];
                                m_nodes[other[i].child(j).id()].m_parent = &m_nodes[i];
                            }
                        }
                    }
                }
                else
                {
                    for (size_type i = 0; i < other.size(); ++i)
                    {
                        f(m_nodes[i].m_data, other[i](), other2[i](), std::forward<Args>(args)...);
                    }
                }
            }
            initialise_level_indexing();
        }

        template <typename U, typename Alloc, typename Func, typename... Args>
        void construct_from_ntree(const ntree<U, Alloc> &tree, Func &&f, Args &&...args)
        {
            // clear the tree if it is not already empty
            if (!empty())
            {
                CALL_AND_HANDLE(clear(), "Failed to construct tree object from an ntree object.  Unable to clear the current tree.");
            }

            if (!tree.empty())
            {
                size_type nleaves = 0;
                m_nleaves = tree.nleaves();
                m_nodes.resize(tree.size());
                size_type index = 0;
                m_nodes[0].m_parent = nullptr;
                for (auto it = tree.begin(); it != tree.end(); ++it)
                {
                    m_nodes[index].m_id = index;
                    f(m_nodes[index].m_data, it->data(), std::forward<Args>(args)...);

                    m_nodes[index].m_level = it->level();
                    m_nodes[index].m_children.resize(it->size());
                    if (it->empty())
                    {
                        m_nodes[index].m_lid = nleaves;
                        ++nleaves;
                    }
                    else
                    {
                        m_nodes[index].m_lid = m_nleaves;
                    }

                    if (!it->empty())
                    {
                        size_type child_index = 0;
                        size_type index2 = index;
                        for (auto jt = it; jt != tree.end(); ++jt)
                        {
                            if (&(*jt) == &(it->operator[](child_index)))
                            {
                                m_nodes[index].m_children[child_index] = &m_nodes[index2];

                                m_nodes[index2].m_index = m_nodes[index].m_index;
                                m_nodes[index2].m_index.push_back(child_index);

                                m_nodes[index2].m_child_id = child_index;
                                m_nodes[index2].m_parent = &m_nodes[index];
                                ++child_index;
                            }
                            if (child_index == it->size())
                            {
                                break;
                            }
                            ++index2;
                        }
                    }
                    ++index;
                }
            }
            initialise_level_indexing();
        }

    protected:
        void initialise_level_indexing()
        {
            // size_type max_level = 0;
            // for(size_type i = 0; i  < m_nodes.size(); ++i)
            //{
            //     if(m_nodes[i].m_level > max_level){max_level = m_nodes[i].m_level;}
            // }
            // std::vector<size_type> nper_level(max_level+1); std::fill(nper_level.begin(), nper_level.end(), 0);
            // m_level_indexing.resize(max_level+1);

            // for(size_type i = 0; i  < m_nodes.size(); ++i)
            //{
            //     ++nper_level[m_nodes[i].m_level];
            // }
            // for(size_type i = 0; i < m_level_indexing.size(); ++i)
            //{
            //     m_level_indexing.resize(nper_level[i]);
            // }
            // std::fill(nper_level.begin(), nper_level.end(), 0);
            // for(size_type i = 0; i  < m_nodes.size(); ++i)
            //{
            //     m_level_indexing[m_nodes[i].m_level][nper_level[m_nodes[i].m_level]] = i;
            //     ++nper_level[m_nodes[i].m_level];
            // }
        }

    private:
        template <typename U>
        typename std::enable_if<!std::is_same<U, T>::value, bool>::type same_reference(const tree_base<U> & /* other */) { return false; }

        template <typename U>
        typename std::enable_if<std::is_same<U, T>::value, bool>::type same_reference(const tree_base<U> &other) { return this == &other; }
    };

    template <typename T>
    class tree : public tree_base<T>
    {
    public:
        using base_type = tree_base<T>;

        using tree_reference = typename base_type::tree_reference;
        using const_tree_reference = typename base_type::const_tree_reference;

        using node_container_type = typename base_type::node_container_type;

        using iterator = typename node_container_type::iterator;
        using const_iterator = typename node_container_type::const_iterator;
        using reverse_iterator = typename node_container_type::reverse_iterator;
        using const_reverse_iterator = typename node_container_type::const_reverse_iterator;

    public:
        template <typename... Args>
        tree(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (...)
        {
            throw;
        }

        template <typename... Args>
        tree_reference operator=(Args &&...args) { CALL_AND_RETHROW(return base_type::operator=(std::forward<Args>(args)...)); }

    public:
        ////return iterators over the child nodes
        iterator begin() { return iterator(tree_base<T>::m_nodes.begin()); }
        iterator end() { return iterator(tree_base<T>::m_nodes.end()); }
        const_iterator begin() const { return const_iterator(tree_base<T>::m_nodes.begin()); }
        const_iterator end() const { return const_iterator(tree_base<T>::m_nodes.end()); }

        reverse_iterator rbegin() { return reverse_iterator(tree_base<T>::m_nodes.rbegin()); }
        reverse_iterator rend() { return reverse_iterator(tree_base<T>::m_nodes.rend()); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(tree_base<T>::m_nodes.rbegin()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(tree_base<T>::m_nodes.rend()); }
    };

    template <typename T, typename U, typename = typename std::enable_if<is_tree<U>::value && is_tree<T>::value, void>::type>
    static inline bool
    has_same_structure(const T &t, const U &u)
    {
        if (t.size() != u.size())
        {
            return false;
        }
        if (t.nleaves() != u.nleaves())
        {
            return false;
        }
        for (typename T::size_type i = 0; i < t.size(); ++i)
        {
            if (!node_has_same_structure(t[i], u[i]))
            {
                return false;
            }
        }
        return true;
    }

    // function for checking if a tree t has  size compatible with the trees u
    template <typename T, typename U>
    static inline typename std::enable_if<is_tree<U>::value && is_tree<T>::value, bool>::type
    has_compatible_size(const T &t, const U &u)
    {
        using traits = node_data_traits::size_comparison_traits<T, U>;
        static_assert(traits::is_applicable::value, "Failed to check if the two trees have compatible sizes.  The size comparison operator is not applicable with the input type.");
        traits f;

        if (!has_same_structure(t, u))
        {
            return false;
        }
        for (typename T::size_type i = 0; i < t.size(); ++i)
        {
            if (!f(t[i](), u[i]()))
            {
                return false;
            }
        }
        return true;
    }

    template <typename T, typename U, typename Func>
    static inline typename std::enable_if<is_tree<U>::value && is_tree<T>::value, bool>::type
    evaluate_condition_on_nodes(const T &t, const U &u, Func &&f)
    {
        if (!has_same_structure(t, u))
        {
            return false;
        }
        for (typename T::size_type i = 0; i < t.size(); ++i)
        {
            if (!f(t[i](), u[i]()))
            {
                return false;
            }
        }
        return true;
    }

    // function for checking if a tree t has  size compatible with the two trees u, v for trees which are sized using two trees.
    template <typename T, typename U, typename V>
    static inline typename std::enable_if<is_tree<U>::value && is_tree<T>::value && is_tree<V>::value, bool>::type
    has_compatible_size(const T &t, const U &u, const V &v)
    {
        using traits = node_data_traits::size_comparison_traits<T, U, V>;
        static_assert(traits::is_applicable::value, "Failed to check if the two trees have compatible sizes.  The size comparison operator is not applicable with the input type.");
        traits f;

        if (!has_same_structure(t, u) || !has_same_structure(t, v))
        {
            return false;
        }
        for (typename T::size_type i = 0; i < t.size(); ++i)
        {
            if (!f(t[i](), u[i](), v[i]()))
            {
                return false;
            }
        }
        return true;
    }

    template <typename T, typename U, typename V, typename Func>
    static inline typename std::enable_if<is_tree<U>::value && is_tree<T>::value && is_tree<V>::value, bool>::type
    evaluate_condition_on_nodes(const T &t, const U &u, const V &v, Func &&f)
    {

        if (!has_same_structure(t, u) || !has_same_structure(t, v))
        {
            return false;
        }
        for (typename T::size_type i = 0; i < t.size(); ++i)
        {
            if (!f(t[i](), u[i](), v[i]()))
            {
                return false;
            }
        }
        return true;
    }

    // template <typename T>
    // std::ostream& operator<<(std::ostream& os, const tree<T>& t){os << "tree : {" << std::endl; for(const auto& i : t){os << i << std::endl;}    os << "}" << std::endl;return os;}

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const tree<T> &t)
    {
        os << "tree : " << t.root() << ";" << std::endl;
        return os;
    }
} // namespace ttns

#endif // PYTTN_TTNS_LIB_TTN_TREE_TREE_HPP_//
