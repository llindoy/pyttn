///
/// @file ntree_node.hpp
/// @author Lachlan Lindoy
/// @date 14/08/2018
/// @version 1.0
/// 
/// @brief Interfaces for the ntree class used for constructing the topology of the multilayer multiconfiguration time-depedent hartree wavefunction
/// 
/// This file contains the definitions of the ntree required for setting up the hierarchy of the ml-mctdh wavefunction.  This is a general purpose tree 
/// implementation which supports an arbitrary number of children per node.  
///

#ifndef TTNS_DATASTRUCTURES_NTREE_NODE_HPP
#define TTNS_DATASTRUCTURES_NTREE_NODE_HPP

#include "ntree_forward_decl.hpp"
#include <common/exception_handling.hpp>

#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <iterator>

namespace ttns
{

template <typename Tree> 
class ntree_node
{
public:
    using size_type = typename Tree::size_type;
    using difference_type = typename Tree::difference_type;

    using value_type = typename Tree::value_type;
    using reference = value_type&;
    using const_reference = const value_type&;

    using node_type = ntree_node<Tree>;
    using node_reference = node_type& ;
    using const_node_reference = const node_type& ;

    using tree_type = Tree;
    using tree_reference = tree_type&;
    using const_tree_reference = const tree_type&;

    using children_type = std::vector<node_type*, typename tree_type::node_pointer_allocator_type>;

    friend class ntree<value_type, typename tree_type::allocator_type>;

    using child_iterator = typename children_type::iterator;
    using const_child_iterator = typename children_type::const_iterator;
    using reverse_child_iterator = typename children_type::reverse_iterator;
    using const_reverse_child_iterator = typename children_type::const_reverse_iterator;

    template <typename iterator>
    class ntree_node_child_iterator
    {
    public:
        typedef iterator base_iterator_type;
        typedef std::random_access_iterator_tag iterator_type;
        typedef node_type value_type;
        typedef typename tmp::choose<tmp::is_const_iterator<iterator>::value, const node_type*, node_type*>::type pointer;
        typedef typename tmp::choose<tmp::is_const_iterator<iterator>::value, const node_type&, node_type&>::type reference;

        typedef ntree_node_child_iterator self_type;

        //determine the const/non-const version of the iterator so that we can implement the conversion function
        typedef typename tmp::choose<
                tmp::is_reverse_iterator<iterator>::value, 
                typename tmp::choose<tmp::is_const_iterator<iterator>::value, reverse_child_iterator, const_reverse_child_iterator>::type, 
                typename tmp::choose<tmp::is_const_iterator<iterator>::value, child_iterator, const_child_iterator>::type>::type  convertible_iterator;

    private:
        base_iterator_type m_iter;

    public:
        ntree_node_child_iterator() {}
        ~ntree_node_child_iterator() {}
        ntree_node_child_iterator(const self_type& other) : m_iter(other.m_iter) {}
        ntree_node_child_iterator(const base_iterator_type& src) : m_iter(src) {}

        self_type& operator=(const self_type& other)
        {
            if(this == &other){return *this;}
            m_iter = other.m_iter;
            return *this;
        }

        self_type& operator=(const base_iterator_type& src){m_iter = src;   return *this;}

        base_iterator_type base() const{return m_iter;}

        operator ntree_node_child_iterator<convertible_iterator>() const {return ntree_node_child_iterator<convertible_iterator>(m_iter);}

        reference operator*() const{return **m_iter;}
        pointer operator->() const{ return *m_iter;}
        reference operator[](const difference_type& n){return **(m_iter+n);}

        self_type& operator++(){++ m_iter;  return *this;}
        self_type operator++(int){self_type ret(*this); ++m_iter;   return ret;}

        self_type& operator--(){--m_iter;   return *this;}
        self_type operator--(int){self_type ret(*this); --m_iter;   return ret;}

        self_type& operator+=(const difference_type& n) {this->m_iter += n; return *this;}
        self_type& operator-=(const difference_type& n) {this->m_iter -= n; return *this;}

        self_type operator+(const difference_type& n) const {return self_type(this->m_iter + n);}
        self_type operator-(const difference_type& n) const {return self_type(this->m_iter - n);}

        difference_type operator-(const self_type& s) const {return this->m_iter - s.m_iter;}

        bool operator==(const self_type& rhs) const { return this->m_iter == rhs.m_iter;}
        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }
        bool operator<(const self_type& rhs) const {return this->m_iter < rhs.m_iter;}
        bool operator<=(const self_type& rhs) const {return this->m_iter <= rhs.m_iter;}
        bool operator>(const self_type& rhs) const {return this->m_iter > rhs.m_iter;}
        bool operator>=(const self_type& rhs) const {return this->m_iter >= rhs.m_iter;}
    };  

    //iterators over the children of the ntree_node
    typedef ntree_node_child_iterator<child_iterator> iterator;
    typedef ntree_node_child_iterator<const_child_iterator> const_iterator;
    typedef ntree_node_child_iterator<reverse_child_iterator> reverse_iterator;
    typedef ntree_node_child_iterator<const_reverse_child_iterator> const_reverse_iterator;

    friend class ntree_builder<value_type>;
protected:
    tree_type* m_tree;
    node_type* m_parent;
    children_type m_children;
    value_type m_data;
    size_type m_size;
    size_type m_nleaves;
    size_type m_level;


protected:
    void decrement_level()
    {
        ASSERT(m_level != 0, "Cannot decrement level of root.");
        --m_level;
        for(auto& ch : m_children)
        {
            CALL_AND_RETHROW(ch->decrement_level());
        }
    }

protected:
    bool uninitialised() const{return (m_parent == nullptr && m_tree == nullptr);}

    ntree_node(const ntree_node& node) : m_tree(node->m_node), m_parent(node->m_parent), m_children(node->m_children), m_data(node->m_data), m_size(node->m_size), m_nleaves(node->m_nleaves), m_level(0){}
    ntree_node(const value_type& val) : m_tree(nullptr), m_parent(nullptr), m_children(), m_data(val), m_size(1), m_nleaves(1), m_level(0){}

    ntree_node& operator=(const ntree_node& other)
    {
        clear();
        m_data = other.m_data;
        m_size = other.m_size;
        m_nleaves = other.m_nleaves;
        m_level = other.m_level;
        for(auto & ch : other.m_children)
        {
            node_type* n = ch->copy_to_tree(*m_tree);
            n->m_parent = this;
            m_children.push_back(n);
            n = nullptr;
        }
        return *this;
    }   


    node_type* copy_to_tree(tree_type& _tree) const
    {
        node_type* n = _tree.create_node();
        n->m_data = this->m_data; 
        n->m_size = this->m_size;
        n->m_tree = &_tree;
        n->m_nleaves = this->m_nleaves;
        n->m_level = this->m_level;
    
        for(auto & ch : m_children)
        {
            node_type* q = ch->copy_to_tree(_tree);
            q->m_parent = n;
            n->m_children.push_back(q);
            q = nullptr;
        }
        return n;
    }

    void clear_children()
    {
        for(auto & ch : m_children)
        {
            ch->clear_children();
            m_tree->destroy_node(ch);
            ch = nullptr;
        }
        m_children.clear();
    }

public:
    bool operator==(const ntree_node& o) const
    {
        return (m_size == o.m_size && m_level=o.m_level && m_parent ==o.m_parent && m_tree = o.m_tree);
    }

public:
    ntree_node() : m_tree(nullptr), m_parent(nullptr), m_children(), m_data(), m_size(1), m_nleaves(1), m_level(0){}

    ~ntree_node()
    {
        if(!uninitialised()){clear_children();}
        m_size = 0;
        m_level = 0;
        m_tree = nullptr;
        m_parent = nullptr;
    }

    size_type level() const{return m_level;}
    size_type nleaves() const{return m_nleaves;}
    size_type size() const{return m_children.size();}
    size_type subtree_size() const{return m_size;}

    bool empty() const{return m_children.empty();}
    bool is_root() const{return m_parent == nullptr;}
    bool is_leaf() const{return m_children.size() == 0;}

    const value_type& operator()() const{return m_data;}
    value_type& operator()(){return m_data;}
    const value_type& value() const{return m_data;}
    value_type& value(){return m_data;}
    const value_type& data() const{return m_data;}
    value_type& data(){return m_data;}

    tree_type& tree()
    {
        ASSERT(m_tree != nullptr, "Unable to find tree.  The node is not associated with a tree.");
        return *m_tree;
    }
    
    node_type& parent() 
    {
        ASSERT(!is_root(), "Unable to access parent.  The node has no parent.");
        return *m_parent;
    }

    node_type& at(size_type n)
    {
        ASSERT(n < m_children.size(), "Unable to access element of tree.  Element is out of bounds.");
        return *(this->m_children[n]);
    }
    const node_type& at(size_type n) const
    {
        ASSERT(n < m_children.size(), "Unable to access element of tree.  Element is out of bounds.");
        return *(this->m_children[n]);
    }

    node_type& operator[](size_type n)
    {
        ASSERT(n < m_children.size(), "Unable to access element of tree.  Element is out of bounds.");
        return *(this->m_children[n]);
    }
    const node_type& operator[](size_type n) const
    {
        ASSERT(n < m_children.size(), "Unable to access element of tree.  Element is out of bounds.");
        return *(this->m_children[n]);
    }

    node_type& at(const std::vector<size_type>& inds, size_type index=0)
    {
        if(inds.size() == 0){return *this;}
        ASSERT(index < inds.size(), "Invalid index argument.");
        ASSERT(inds[index] < m_children.size(), "Unable to access element of tree.  Element is out of bounds.");
        if(index +1  == inds.size())
        {
            CALL_AND_RETHROW(return this->operator[](inds[index]));
        }
        else
        {
            CALL_AND_RETHROW(return this->operator[](inds[index]).at(inds, index+1));
        }
    }

    const node_type& at(const std::vector<size_type>& inds, size_type index=0) const
    {
        if(inds.size() == 0){return *this;}
        ASSERT(index < inds.size(), "Invalid index argument.");
        ASSERT(inds[index] < m_children.size(), "Unable to access element of tree.  Element is out of bounds.");

        if(index +1  == inds.size())
        {
            CALL_AND_RETHROW(return this->operator[](inds[index]));
        }
        else
        {
            CALL_AND_RETHROW(return this->operator[](inds[index]).at(inds, index+1));
        }
    }

    node_type& back()
    {
        ASSERT(!empty(), "Unable to access the final child of the tree. Node has no children");
        return *(this->m_children.back());
    }
    const node_type& back() const
    {
        ASSERT(!empty(), "Unable to access the final child of the tree. Node has no children");
        return *(this->m_children.back());
    }
    node_type& front()
    {
        ASSERT(!empty(), "Unable to access the first child of the tree. Node has no children");
        return *(this->m_children.front());
    }
    const node_type& front() const
    {
        ASSERT(!empty(), "Unable to access the first child of the tree. Node has no children");
        return *(this->m_children.front());
    }

    void clear()
    {
        size_type _size = m_size-1;
        size_type _leaves = m_nleaves-1;
        clear_children();
        node_type* q = static_cast<node_type*>(this);
        while(!q->is_root())
        {
            q = q->m_parent;
            q->m_size -= _size;
            q->m_nleaves -= _leaves;
        }
        m_size = 1;
        m_nleaves = 1;
    }

    size_type insert(const node_type& src)
    {
        ASSERT(m_tree != nullptr, "Failed to add child to node.  The node is not associated with a tree.");
        ASSERT(src.m_tree != m_tree, "Failed to add child to node.  The operation would introduce a cycle.");

        node_type* q = static_cast<node_type*>(this);
        node_type* p = src.copy_to_tree(*m_tree);
        p->m_parent = this;
        p->m_level = this->m_level + 1;

        size_type additional_leaves = p->m_nleaves;
        if(m_children.size() == 0){--additional_leaves;}

        m_size += p->m_size;
        m_nleaves += additional_leaves;
        while(!q->is_root())
        {
            q = q->m_parent;
            q->m_size += p->m_size;
            q->m_nleaves += additional_leaves;
        }
        q = nullptr;

        size_type ind = m_children.size();
        m_children.push_back(p);
        return ind;
    }  

    size_type insert(const value_type& src = value_type())
    {
        ASSERT(m_tree != nullptr, "Failed to add child to node.  The node is not associated with a tree.");

        node_type* q = static_cast<node_type*>(this);

        node_type* p = m_tree->create_node();
        p->m_tree = m_tree;
        p->m_parent = this;
        p->m_data = src;
        p->m_size = 1;
        p->m_nleaves = 1;
        p->m_level = this->m_level + 1;
        size_type additional_leaves = p->m_nleaves;
        if(m_children.size() == 0){--additional_leaves;}

        m_size += p->m_size;
        m_nleaves += additional_leaves;
        while(!q->is_root())
        {
            q = q->m_parent;
            q->m_size += p->m_size;
            q->m_nleaves += additional_leaves;
        }

        size_type ind = m_children.size();
        m_children.push_back(p);
        p = nullptr;
        q = nullptr;
        return ind;
    }

    size_type insert_front(const node_type& src)
    {
        ASSERT(m_tree != nullptr, "Failed to add child to node.  The node is not associated with a tree.");
        ASSERT(src.m_tree != m_tree, "Failed to add child to node.  The operation would introduce a cycle.");

        node_type* q = static_cast<node_type*>(this);
        node_type* p = src.copy_to_tree(*m_tree);
        p->m_parent = this;
        p->m_level = this->m_level + 1;

        size_type additional_leaves = p->m_nleaves;
        if(m_children.size() == 0){--additional_leaves;}

        m_size += p->m_size;
        m_nleaves += additional_leaves;
        while(!q->is_root())
        {
            q = q->m_parent;
            q->m_size += p->m_size;
            q->m_nleaves += additional_leaves;
        }
        q = nullptr;

        size_type ind = m_children.size();
        m_children.insert(m_children.begin(), p);
        return 0;
    }  

    size_type insert_front(const value_type& src = value_type())
    {
        ASSERT(m_tree != nullptr, "Failed to add child to node.  The node is not associated with a tree.");

        node_type* q = static_cast<node_type*>(this);

        node_type* p = m_tree->create_node();
        p->m_tree = m_tree;
        p->m_parent = this;
        p->m_data = src;
        p->m_size = 1;
        p->m_nleaves = 1;
        p->m_level = this->m_level + 1;
        size_type additional_leaves = p->m_nleaves;
        if(m_children.size() == 0){--additional_leaves;}

        m_size += p->m_size;
        m_nleaves += additional_leaves;
        while(!q->is_root())
        {
            q = q->m_parent;
            q->m_size += p->m_size;
            q->m_nleaves += additional_leaves;
        }

        size_type ind = m_children.size();
        m_children.insert(m_children.begin(), p);
        p = nullptr;
        q = nullptr;
        return 0;
    }


protected:
    void remove_child(child_iterator it)
    {
        size_type _size = (*it)->m_size;
        size_type leaves_to_remove = (*it)->m_nleaves;
        if(m_children.size() == 1){--leaves_to_remove;}

        m_size -= _size;
        m_nleaves -= leaves_to_remove;
        node_type* q = static_cast<node_type*>(this);
        while(!q->is_root())
        {
            q = q->m_parent;
            q->m_size -= _size;
            q->m_nleaves -= leaves_to_remove;
        }

        (*it)->clear_children();
        m_tree->destroy_node(*it);
        m_children.erase(it);
    }

public:
    void remove(size_type ind)
    {
        ASSERT(m_tree != nullptr, "Failed to remove child from node.  The node is not associated with a tree.");

        ASSERT(ind < m_children.size(), "Failed to remove child from node.  Index out of bounds")

        child_iterator it = m_children.begin()  + ind;
        remove_child(it);
    }

    void remove(const node_type& src)
    {
        ASSERT(m_tree != nullptr, "Failed to remove child from node.  The node is not associated with a tree.");
        ASSERT(src.m_tree == m_tree, "Faileud to remove child from node.  The node to be removed is not associated with the same tree.");

        auto it = std::find(m_children.begin(), m_children.end(), src);
        if(it == m_children.end()){return;}

        remove_child(it);
    }

    
    std::ostream& as_json(std::ostream& os) const
    {
        os << "{";
        os << "\"data\": " << m_data ;
        if(size() > 0)
        {
            os << ", \"children\": [";
            for(size_t i=0; i<size(); ++i)
            {
                m_children[i]->as_json(os);
                if(i+1 != size())
                {
                    os << ",";
                }
            }
            os << "]";
        }
        os << "}";
    
        return os;
    }

public:
    ////return iterators over the child nodes 
    iterator begin() {  return iterator(m_children.begin());  }
    iterator end() {  return iterator(m_children.end());  }
    const_iterator begin() const {  return const_iterator(m_children.begin());  }
    const_iterator end() const {  return const_iterator(m_children.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_children.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_children.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_children.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_children.rend());  }

};  //class ntree_node

template <typename Tree> 
std::ostream& operator<<(std::ostream& os, const ntree_node<Tree>& t)
{
    os << "(";
    os << t();
    for(size_t i=0; i<t.size(); ++i)
    {
        os << t[i];
        //if(i+1 != t.size()){os << ",";}
    }
    os << ")";
    return os;
}

}   //namespace ttns


#endif  //  TTNS_DATASTRUCTURES_NTREE_NODE_HPP    //

