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

#ifndef PYTTN_TTNS_LIB_TTN_TREE_TREE_NODE_HPP_
#define PYTTN_TTNS_LIB_TTN_TREE_TREE_NODE_HPP_

#include <common/exception_handling.hpp>
#include <common/tmp_funcs.hpp>

#include "tree_forward_decl.hpp"
#include "node_data_traits.hpp"
#include "../../utils/serialisation.hpp"

#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <iterator>
#include <list>

namespace ttns
{

    template <typename Tree>
    class tree_node_base : public tree_node_tag
    {
    public:
        // important typedefs
        using tree_type = typename Tree::tree_type;
        using tree_reference = typename tree_type::tree_reference;
        using const_tree_reference = typename tree_type::const_tree_reference;
        using size_type = typename tree_type::size_type;
        using difference_type = typename tree_type::difference_type;
        using value_type = typename tree_type::value_type;
        using reference = typename tree_type::reference;
        using const_reference = typename tree_type::const_reference;
        using node_type = typename tree_type::node_type;
        using node_reference = typename tree_type::node_reference;
        using const_node_reference = typename tree_type::const_node_reference;
        using children_type = std::vector<node_type *>;

        // friendship classes
        friend class tree_base<value_type>;
        friend class tree<value_type>;

    public:
        using child_iterator = typename children_type::iterator;
        using const_child_iterator = typename children_type::const_iterator;
        using reverse_child_iterator = typename children_type::reverse_iterator;
        using const_reverse_child_iterator = typename children_type::const_reverse_iterator;

        template <typename iterator>
        class tree_node_child_iterator
        {
        public:
            using base_iterator_type = iterator;
            using iterator_type = typename std::iterator_traits<iterator>::iterator_category;
            using value_type = node_type;

            using pointer = typename tmp::choose<tmp::is_const_iterator<iterator>::value, const node_type *, node_type *>::type;
            using reference = typename tmp::choose<tmp::is_const_iterator<iterator>::value, const node_type &, node_type &>::type;

            using self_type = tree_node_child_iterator;

            // determine the const/non-const version of the iterator so that we can implement the conversion function
            using convertible_iterator = typename tmp::choose<
                tmp::is_reverse_iterator<iterator>::value,
                const_reverse_child_iterator,
                const_child_iterator>::type;

        private:
            base_iterator_type m_iter;

        public:
            tree_node_child_iterator() {}
            ~tree_node_child_iterator() {}
            tree_node_child_iterator(const self_type &other) : m_iter(other.m_iter) {}
            tree_node_child_iterator(const base_iterator_type &src) : m_iter(src) {}

            self_type &operator=(const self_type &other)
            {
                if (this == &other)
                {
                    return *this;
                }
                m_iter = other.m_iter;
                return *this;
            }
            self_type &operator=(const base_iterator_type &src)
            {
                m_iter = src;
                return *this;
            }

            base_iterator_type base() const { return m_iter; }
            // operator tree_node_child_iterator<convertible_iterator>() const {return tree_node_child_iterator<convertible_iterator>(m_iter);}

            reference operator*() const { return **m_iter; }
            pointer operator->() const { return *m_iter; }
            reference operator[](const difference_type &n) { return **(m_iter + n); }

            self_type &operator++()
            {
                ++m_iter;
                return *this;
            }
            self_type operator++(int)
            {
                self_type ret(*this);
                ++m_iter;
                return ret;
            }
            self_type &operator--()
            {
                --m_iter;
                return *this;
            }
            self_type operator--(int)
            {
                self_type ret(*this);
                --m_iter;
                return ret;
            }
            self_type &operator+=(const difference_type &n)
            {
                m_iter += n;
                return *this;
            }
            self_type &operator-=(const difference_type &n)
            {
                m_iter -= n;
                return *this;
            }
            self_type operator+(const difference_type &n) const { return self_type(m_iter + n); }
            self_type operator-(const difference_type &n) const { return self_type(m_iter - n); }

            difference_type operator-(const self_type &s) const { return m_iter - s.m_iter; }

            bool operator==(const self_type &rhs) const { return m_iter == rhs.m_iter; }
            bool operator!=(const self_type &rhs) const { return !(*this == rhs); }
            bool operator<(const self_type &rhs) const { return m_iter < rhs.m_iter; }
            bool operator<=(const self_type &rhs) const { return m_iter <= rhs.m_iter; }
            bool operator>(const self_type &rhs) const { return m_iter > rhs.m_iter; }
            bool operator>=(const self_type &rhs) const { return m_iter >= rhs.m_iter; }
        };

        // iterators over the children of the tree_node
        using iterator = tree_node_child_iterator<child_iterator>;
        using const_iterator = tree_node_child_iterator<const_child_iterator>;
        using reverse_iterator = tree_node_child_iterator<reverse_child_iterator>;
        using const_reverse_iterator = tree_node_child_iterator<const_reverse_child_iterator>;

    protected:
        value_type m_data;
        tree_type *m_tree;
        node_type *m_parent;
        children_type m_children;

        node_data_traits::clear_traits<value_type> m_clear;
        size_type m_id;
        size_type m_child_id;
        size_type m_level;
        size_type m_lid;
        std::list<size_type> m_index;

        void clear()
        {
            for (auto &ch : m_children)
            {
                ch = nullptr;
            }
            m_parent = nullptr;
            m_clear(m_data);
        }

        tree_node_base &operator=(const tree_node_base &other) = delete;

    public:
        tree_node_base() : m_parent(nullptr), m_clear(), m_id(0), m_child_id(0), m_level(0) {}
        tree_node_base(tree_node_base &&other) : m_clear() { operator=(std::forward<tree_node_base>(other)); }
        ~tree_node_base()
        {
            m_parent = nullptr;
            m_tree = nullptr;
        }

        tree_node_base &operator=(tree_node_base &&other)
        {
            m_parent = other.m_parent;
            other.m_parent = nullptr;
            m_children = std::move(other.m_children);
            m_id = std::move(other.m_id);
            other.m_id = 0;
            m_child_id = std::move(other.m_child_id);
            other.m_child_id = 0;
            m_level = std::move(other.m_level);
            m_index = std::move(other.m_index);
            m_data = std::move(other.m_data);
            return *this;
        }

    protected:
#ifdef CEREAL_LIBRARY_FOUND
        friend class serialisation_node_save_wrapper<node_type, size_type>;
        friend class serialisation_node_load_wrapper<node_type, size_type>;

        tree_node_base &operator=(const serialisation_node_load_wrapper<node_type, size_type> &other)
        {
            m_id = other.m_node.m_id;
            m_child_id = other.m_node.m_child_id;
            m_level = other.m_node.m_level;
            m_data = other.m_node.m_data;
            m_index = other.m_node.m_index;
            return *this;
        }

        tree_node_base &operator=(serialisation_node_load_wrapper<node_type, size_type> &&other)
        {
            return this->operator=(std::move(other.m_node));
        }
#endif

    protected:
        void set_leaf_index(size_type lid) { m_lid = lid; }

    public:
        const_reference operator()() const { return m_data; }
        reference operator()() { return m_data; }
        const_reference value() const { return m_data; }
        reference value() { return m_data; }
        const_reference data() const { return m_data; }
        reference data() { return m_data; }

        size_type size() const { return m_children.size(); }
        bool empty() const { return m_children.empty(); }
        bool is_root() const { return m_parent == nullptr; }
        bool is_leaf() const { return m_children.size() == 0; }
        size_type leaf_index() const
        {
            ASSERT(m_children.size() == 0, "Failed to return leaf index of the node.  The requested node is not a leaf.");
            return m_lid;
        }

        bool is_child_of(size_t i) const
        {
            if (!is_root())
            {
                return m_parent->m_id == i;
            }
            else
            {
                return false;
            }
        }

        node_reference parent()
        {
            ASSERT(!is_root(), "Unable to return a reference to the nodes parent.  The node has no parent.");
            return *m_parent;
        }
        const_node_reference parent() const
        {
            ASSERT(!is_root(), "Unable to return a reference to the nodes parent.  The node has no parent.");
            return *m_parent;
        }
        node_type const *parent_pointer() const
        {
            ASSERT(!is_root(), "Unable to return a reference to the nodes parent.  The node has no parent.");
            return m_parent;
        }
        node_reference at(size_type n)
        {
            ASSERT(n < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            return *(m_children[n]);
        }
        const_node_reference at(size_type n) const
        {
            ASSERT(n < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            return *(m_children[n]);
        }
        node_reference front()
        {
            ASSERT(m_children.size() != 0, "Unable to access the front child node.  This node has no children.");
            return *(this->m_children.front());
        }
        const_node_reference front() const
        {
            ASSERT(m_children.size() != 0, "Unable to access the front child node.  This node has no children.");
            return *(this->m_children.front());
        }
        node_reference back()
        {
            ASSERT(m_children.size() != 0, "Unable to access the back child node.  This node has no children.");
            return *(this->m_children.back());
        }
        const_node_reference back() const
        {
            ASSERT(m_children.size() != 0, "Unable to access the back child node.  This node has no children.");
            return *(this->m_children.back());
        }
        node_reference operator[](size_type n)
        {
            ASSERT(n < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            return *(this->m_children[n]);
        }
        const_node_reference operator[](size_type n) const
        {
            ASSERT(n < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            return *(this->m_children[n]);
        }
        node_reference child(size_type n)
        {
            ASSERT(n < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            return *(this->m_children[n]);
        }
        const_node_reference child(size_type n) const
        {
            ASSERT(n < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            return *(this->m_children[n]);
        }
        node_type const *child_pointer(size_type n) const
        {
            ASSERT(n < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            return this->m_children[n];
        }

        node_reference at(const std::list<size_type> &tree_index)
        {
            CALL_AND_RETHROW(return at(tree_index.begin(), tree_index.end()));
        }
        const_node_reference at(const std::list<size_type> &tree_index) const
        {
            CALL_AND_RETHROW(return at(tree_index.begin(), tree_index.end()));
        }
        size_type id_at(const std::list<size_type> &tree_index) const
        {
            CALL_AND_RETHROW(return id_at(tree_index.begin(), tree_index.end()));
        }

        template <typename IT>
        node_reference at(IT first, IT last)
        {
            ASSERT(*first < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            size_type ind = *first;
            ++first;
            if (first != last)
            {
                CALL_AND_HANDLE(return (this->m_children[ind]->at(first, last)), "Failed to return node by tree index.");
            }
            else
            {
                return *(this->m_children[ind]);
            }
        }
        template <typename IT>
        const_node_reference at(IT first, IT last) const
        {
            ASSERT(*first < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            size_type ind = *first;
            ++first;
            if (first != last)
            {
                CALL_AND_HANDLE(return (this->m_children[ind]->at(first, last)), "Failed to return node by tree index.");
            }
            else
            {
                return *(this->m_children[ind]);
            }
        }
        template <typename IT>
        size_type id_at(IT first, IT last) const
        {
            ASSERT(*first < m_children.size(), "Unable to access child of node.  Index is out of bounds.");
            size_type ind = *first;
            ++first;
            if (first != last)
            {
                CALL_AND_HANDLE(return (this->m_children[ind]->id_at(first, last)), "Failed to return node by tree index.");
            }
            else
            {
                return this->m_children[ind]->id();
            }
        }

        const std::list<size_type> index() const noexcept { return m_index; }
        size_type id() const noexcept { return m_id; }
        size_type child_id() const noexcept { return m_child_id; }
        size_type level() const noexcept { return m_level; }
        const children_type &children() const noexcept { return m_children; }

    public:
        // return iterators over the child nodes
        iterator begin() { return iterator(m_children.begin()); }
        iterator end() { return iterator(m_children.end()); }
        const_iterator begin() const { return const_iterator(m_children.begin()); }
        const_iterator end() const { return const_iterator(m_children.end()); }

        reverse_iterator rbegin() { return reverse_iterator(m_children.rbegin()); }
        reverse_iterator rend() { return reverse_iterator(m_children.rend()); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(m_children.rbegin()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(m_children.rend()); }

#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void save(archive &ar) const
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("index", m_id)), "Failed to serialise tree_node_base object.  Failed to serialise its id.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("local_index", m_child_id)), "Failed to serialise tree_node_base object.  Failed to serialise its local id.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("tree_index", m_index)), "Failed to serialise tree_node_base object.  Failed to serialise its local id.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("level", m_level)), "Failed to serialise tree_node_base object.  Failed to serialise its level.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("data", m_data)), "Failed to serialise tree_node_base object.  Error when serialising the data stored at the node.");
            if (m_children.size() == 0)
            {
                CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_index", m_lid)), "Failed to serialise tree_node_base object.  Failed to serialise its leaf index.");
            }
        }

        template <typename archive>
        void load(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("index", m_id)), "Failed to serialise tree_node_base object.  Failed to serialise its id.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("local_index", m_child_id)), "Failed to serialise tree_node_base object.  Failed to serialise its local id.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("tree_index", m_index)), "Failed to serialise tree_node_base object.  Failed to serialise its local id.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("level", m_level)), "Failed to serialise tree_node_base object.  Failed to serialise its level.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("data", m_data)), "Failed to serialise tree_node_base object.  Error when serialising the data stored at the node.");
            if (m_children.size() == 0)
            {
                CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_index", m_lid)), "Failed to serialise tree_node_base object.  Failed to serialise its leaf index.");
            }
        }
#endif
    }; // class tree_node_base

    template <typename T>
    class tree_node : public tree_node_base<T>
    {
    };

    template <typename T, typename U>
    static inline typename std::enable_if<is_tree_node<U>::value && is_tree_node<T>::value, bool>::type node_has_same_structure(const T &t, const U &u)
    {
        if (t.is_root() != u.is_root())
        {
            return false;
        }
        if (t.size() != u.size())
        {
            return false;
        }
        if (t.id() != u.id())
        {
            return false;
        }
        if (t.level() != u.level())
        {
            return false;
        }
        // if(t.index() != u.index()){return false;}
        if (!t.is_root())
        {
            if (t.child_id() != u.child_id())
            {
                return false;
            }
        }
        return true;
    }

    /*
    template <typename Tree>
    std::ostream& operator<<(std::ostream& os, const tree_node_base<Tree>& t)
    {
        os << "node: " << t.id() << std::endl;
        os << "children: ";
        for(const auto& i : t){os << i.id() << " ";}
        os << std::endl;
        os << t();
        return os;
    }*/

    template <typename Tree>
    std::ostream &operator<<(std::ostream &os, const tree_node_base<Tree> &t)
    {
        os << "(";
        os << t();
        for (size_t i = 0; i < t.size(); ++i)
        {
            os << t[i];
            // if(i+1 != t.size()){os << ",";}
        }
        os << ")";
        return os;
    }

} // namespace ttns

#endif // PYTTN_TTNS_LIB_TTN_TREE_TREE_NODE_HPP_//
