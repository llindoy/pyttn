///
/// @file ntree_iterator.hpp
/// @author Lachlan Lindoy
/// @date 18/08/2018
/// @version 1.0
/// 
/// @brief Interfaces for the ntree class used for constructing the topology of the multilayer multiconfiguration time-depedent hartree wavefunction
/// 
/// This file contains the definitions of the ntree required for setting up the hierarchy of the ml-mctdh wavefunction.  This is a general purpose tree 
/// implementation which supports an arbitrary number of children per node.  
///

#ifndef TTNS_DATASTRUCTURES_NTREE_ITERATORS_HPP
#define TTNS_DATASTRUCTURES_NTREE_ITERATORS_HPP

#include "ntree_forward_decl.hpp"
#include "ntree_node.hpp"
#include <common/exception_handling.hpp>

#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <type_traits>
#include <deque>
#include <stack>

#include <iostream>

namespace ttns
{

template <typename Node> 
class ntree_iterator{};

template <typename Node, bool is_const, typename Alloc>
class ntree_dfs_preorder_iterator
{
public:
    typedef Node node_type;
    typedef typename Alloc::difference_type difference_type;
    typedef std::forward_iterator_tag iterator_category;

    typedef typename tmp::choose<is_const, const Node*, Node*>::type pointer;
    typedef typename tmp::choose<is_const, const Node&, Node&>::type reference;

    typedef ntree_dfs_preorder_iterator self_type;
    typedef typename tmp::choose<is_const, typename node_type::const_iterator, typename node_type::iterator>::type iterator;

    struct stack_elem
    {
        stack_elem() : m_node(nullptr), m_child_iter(), m_visited(false) {}
        stack_elem(const stack_elem& other) : m_node(other.m_node), m_child_iter(other.m_child_iter), m_visited(other.m_visited) {}
        stack_elem(pointer child, const iterator& others, bool visited) : m_node(child), m_child_iter(others), m_visited(visited) { }
        stack_elem& operator=(const stack_elem& other) 
        {
            m_node = other.m_node;
            m_child_iter = other.m_child_iter;
            m_visited = other.m_visited;
        }

        ~stack_elem(){m_node = nullptr;}
        
        bool operator==(const stack_elem& other) const{return (m_node == other.m_node && m_child_iter == other.m_child_iter && m_visited == other.m_visited);}
        bool operator!=(const stack_elem& other) const{return !(m_node == other.m_node && m_child_iter == other.m_child_iter && m_visited == other.m_visited);}

        pointer m_node;
        iterator m_child_iter;
        bool m_visited;
    };

private:
    typedef typename std::allocator_traits<Alloc>::template rebind_alloc<stack_elem>  stack_elem_allocator_type;
    std::stack<stack_elem, std::deque<stack_elem, stack_elem_allocator_type> > m_stack;

public:
    ntree_dfs_preorder_iterator() : m_stack() {}
    ntree_dfs_preorder_iterator(const self_type& other) : m_stack(other.m_stack) {}
    ntree_dfs_preorder_iterator(pointer node){if(node != nullptr){m_stack.push(stack_elem(node, iterator(node->begin()), false));}}
    ~ntree_dfs_preorder_iterator() {}

    self_type& operator=(const self_type& other){m_stack = other.m_stack;}
    bool operator==(const self_type& other) const{return m_stack == other.m_stack;}
    bool operator!=(const self_type& other) const{return m_stack != other.m_stack;}

    reference operator*() const{return *(m_stack.top().m_node);}
    pointer operator->() const{return m_stack.top().m_node;}

    self_type& operator++()
    {
        if(m_stack.empty()){return *this;}

        //this attempts to move down the left most branch of the tree
        if(!m_stack.top().m_visited)
        {
            m_stack.top().m_visited = true;
            if(!m_stack.top().m_node->empty())      
            {
                m_stack.push(stack_elem(&(*m_stack.top().m_child_iter), iterator((m_stack.top().m_child_iter)->begin()), false));
                return *this;
            }
        }

        //now we advance the current element in the stack and pop it if we are finished
        while(true)
        {
            if(!m_stack.top().m_node->empty())
            {
                //advance the child_iterator
                ++(m_stack.top().m_child_iter);
                
                //if we haven't hit the end then we still need to push the child to this and we can continue advancing through the tree
                if(m_stack.top().m_child_iter != m_stack.top().m_node->end())
                {
                    m_stack.push(stack_elem(&(*m_stack.top().m_child_iter), (m_stack.top().m_child_iter)->begin(), false));
                    return *this;
                }
            }
            //otherwise we need to remove the element from the stack
            m_stack.pop();
            //and if we reach an empty stack we are doneR
            if(m_stack.empty()){return *this;}
        }
    }

    self_type operator++(int){self_type ret(*this); ++(*this);  return ret;}

    bool empty() const{return m_stack.empty();}
};  //class ntree_dfs_preorder_iterator


template <typename Node, bool is_const, typename Alloc>
class ntree_dfs_postorder_iterator
{
public:
    typedef Node node_type;
    typedef typename Alloc::difference_type difference_type;
    typedef std::forward_iterator_tag iterator_category;

    typedef typename tmp::choose<is_const, const Node*, Node*>::type pointer;
    typedef typename tmp::choose<is_const, const Node&, Node&>::type reference;

    typedef ntree_dfs_postorder_iterator self_type;
    typedef typename tmp::choose<is_const, typename node_type::const_iterator, typename node_type::iterator>::type iterator;

    struct stack_elem
    {
        stack_elem() : m_node(nullptr), m_child_iter(), m_visited(false) {}
        stack_elem(const stack_elem& other) : m_node(other.m_node), m_child_iter(other.m_child_iter), m_visited(other.m_visited) {}
        stack_elem(pointer child, const iterator& others, bool visited) : m_node(child), m_child_iter(others), m_visited(visited) { }
        stack_elem& operator=(const stack_elem& other) 
        {
            m_node = other.m_node;
            m_child_iter = other.m_child_iter;
            m_visited = other.m_visited;
        }

        ~stack_elem(){m_node = nullptr;}
        
        bool operator==(const stack_elem& other) const{return (m_node == other.m_node && m_child_iter == other.m_child_iter && m_visited == other.m_visited);}
        bool operator!=(const stack_elem& other) const{return !(m_node == other.m_node && m_child_iter == other.m_child_iter && m_visited == other.m_visited);}

        pointer m_node;
        iterator m_child_iter;
        bool m_visited;
    };

private:
    typedef typename std::allocator_traits<Alloc>::template rebind_alloc<stack_elem>  stack_elem_allocator_type;
    std::stack<stack_elem, std::deque<stack_elem, stack_elem_allocator_type> > m_stack;

public:
    ntree_dfs_postorder_iterator() : m_stack() {}
    ntree_dfs_postorder_iterator(const self_type& other) : m_stack(other.m_stack) {}
    ntree_dfs_postorder_iterator(pointer node)
    {
        if(node != nullptr)
        {
            m_stack.push(stack_elem(node, iterator(node->begin()), false));
            //if the node we are currently looking at has children
            while(!m_stack.top().m_node->empty())
            {
                //get the first child of the current top of the stack
                auto stelem = stack_elem(&(*m_stack.top().m_child_iter), (m_stack.top().m_child_iter)->begin(), false);

                //and advance the iterator of the current top of the stack
                ++(m_stack.top().m_child_iter);

                //insert the child of the current node in the stack
                m_stack.push(stelem);
            }
            m_stack.top().m_visited = true;
        }
    }
    ~ntree_dfs_postorder_iterator() {}

    self_type& operator=(const self_type& other){m_stack = other.m_stack;}
    bool operator==(const self_type& other) const{return m_stack == other.m_stack;}
    bool operator!=(const self_type& other) const{return m_stack != other.m_stack;}

    reference operator*() const{return *(m_stack.top().m_node);}
    pointer operator->() const{return m_stack.top().m_node;}

    self_type& operator++()
    {
        if(m_stack.empty()){return *this;}

        while(true)
        {
            //if we are at a leaf node.  Then we have already accessed it so we should remove it from the 
            if(m_stack.top().m_node->empty())
            {
                if(m_stack.top().m_visited)
                {
                    m_stack.pop();
                }
                else
                {
                    m_stack.top().m_visited = true;
                    return *this;
                }
            }
            //if we have accessed all of the children of this node then we see if it has been visited (and if so remove it)
            //otherwise we set that it has been visited and exit.
            else if(m_stack.top().m_child_iter == m_stack.top().m_node->end())
            {
                if(m_stack.top().m_visited)
                {
                    m_stack.pop();
                }
                else
                {
                    m_stack.top().m_visited = true;
                    return *this;
                }
            }
            //here we need to proceed down the tree structure
            else
            {
                while(!m_stack.top().m_node->empty())
                {
                    //get the first child of the current top of the stack
                    auto stelem = stack_elem(&(*m_stack.top().m_child_iter), (m_stack.top().m_child_iter)->begin(), false);

                    //and advance the iterator of the current top of the stack
                    ++(m_stack.top().m_child_iter);

                    //insert the child of the current node in the stack
                    m_stack.push(stelem);
                }
            }
            if(m_stack.empty()){return *this;}
        }
    }

    self_type operator++(int){self_type ret(*this); ++(*this);  return ret;}

    bool empty() const{return m_stack.empty();}
};  //class ntree_dfs_preorder_iterator


template <typename Node, bool is_const, typename Alloc>
class ntree_dfs_pre_post_iterator
{
public:
    using node_type = Node;
    using difference_type = typename Alloc::difference_type;
    using size_type = typename Alloc::size_type;
    using iterator_category = std::forward_iterator_tag;

    using pointer = typename tmp::choose<is_const, const Node*, Node*>::type;
    using reference = typename tmp::choose<is_const, const Node&, Node&>::type;

    using self_type = ntree_dfs_pre_post_iterator;
    using iterator = typename tmp::choose<is_const, typename node_type::const_iterator, typename node_type::iterator>::type;

    struct stack_elem
    {
        stack_elem() : m_node(), m_child_iter(), m_times_visited(0), m_children_visited(0) {}
        stack_elem(const stack_elem& other) : m_node(other.m_node), m_child_iter(other.m_child_iter), m_times_visited(other.m_times_visited), m_children_visited(other.m_children_visited) {}
        stack_elem(pointer child, const iterator& others, size_type times_visited) : m_node(child), m_child_iter(others), m_times_visited(times_visited), m_children_visited(0) { }
        stack_elem& operator=(const stack_elem& other) 
        {
            m_node = other.m_node;
            m_child_iter = other.m_child_iter;
            m_times_visited = other.m_times_visited;
            m_children_visited = other.m_children_visited;
        }

        ~stack_elem() {}
        
        bool operator==(const stack_elem& other) const{return (m_node == other.m_node && m_child_iter == other.m_child_iter && m_times_visited == other.m_times_visited && m_children_visited == other.m_children_visited);}

        bool operator!=(const stack_elem& other) const{return !(m_node == other.m_node && m_child_iter == other.m_child_iter && m_times_visited == other.m_times_visited && m_children_visited == other.m_children_visited);}

        pointer m_node;
        iterator m_child_iter;
        size_type m_times_visited;
        size_type m_children_visited;
    };

private:
    
    using stack_elem_allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<stack_elem>;
    std::stack<stack_elem, std::deque<stack_elem, stack_elem_allocator_type> > m_stack;

public:
    ntree_dfs_pre_post_iterator() : m_stack() {}
    ntree_dfs_pre_post_iterator(const self_type& other) : m_stack(other.m_stack) {}
    ntree_dfs_pre_post_iterator(pointer node){if(node != nullptr){m_stack.push(stack_elem(node, iterator(node->begin()), false));}}

    ~ntree_dfs_pre_post_iterator() {}

    self_type& operator=(const self_type& other){m_stack = other.m_stack;}

    bool operator==(const self_type& other) const{return m_stack == other.m_stack;}
    bool operator!=(const self_type& other) const{return m_stack != other.m_stack;}

    reference operator*() const{return *(m_stack.top().m_node);}
    pointer operator->() const{return m_stack.top().m_node;}

    self_type& operator++()
    {
        if(m_stack.empty()){return *this;}

        //this attempts to move down the left most branch of the tree
        if(m_stack.top().m_times_visited == 0)
        {
            ++(m_stack.top().m_times_visited);
            if(!m_stack.top().m_node->empty())      
            {
                ++(m_stack.top().m_children_visited);
                m_stack.push(stack_elem(&(*m_stack.top().m_child_iter), iterator((m_stack.top().m_child_iter)->begin()), 0));
                return *this;
            }
        }

        //now we advance the current element in the stack and pop it if we are finished
        while(true)
        {
            if(!m_stack.top().m_node->empty())
            {
                if(m_stack.top().m_children_visited+1 == m_stack.top().m_times_visited)
                {
                    //advance the child_iterator
                    ++(m_stack.top().m_child_iter);
                    ++(m_stack.top().m_children_visited);
                    
                    //if we haven't hit the end then we still need to push the child to this and we can continue advancing through the tree
                    if(m_stack.top().m_child_iter != m_stack.top().m_node->end())
                    {
                        m_stack.push(stack_elem(&(*m_stack.top().m_child_iter), (m_stack.top().m_child_iter)->begin(), 0));
                        return *this;
                    }
                }
                else{++(m_stack.top().m_times_visited); return *this;}
            }
            else
            {
                if(m_stack.top().m_times_visited == 1){++(m_stack.top().m_times_visited);   return *this;}
            }
            //otherwise we need to remove the element from the stack
            m_stack.pop();
            //and if we reach an empty stack we are done
            if(m_stack.empty()){return *this;}
        }
    }

    self_type operator++(int){self_type ret(*this); ++(*this);  return ret;}
};  //class ntree_dfs_pre_post_iterator


template <typename Node, bool is_const, typename Alloc>
class ntree_bfs_iterator
{
public:
    typedef Node node_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;


    typedef typename tmp::choose<is_const, const Node*, Node*>::type pointer;
    typedef typename tmp::choose<is_const, const Node&, Node&>::type reference;

    typedef ntree_bfs_iterator self_type;

private:
    typedef typename std::allocator_traits<Alloc>::template rebind_alloc<pointer>  node_pointer_allocator_type;
    std::deque<pointer, node_pointer_allocator_type> m_queue;

public:
    ntree_bfs_iterator() : m_queue() {}
    ntree_bfs_iterator(const self_type& other) : m_queue(other.m_queue) {}
    ntree_bfs_iterator(pointer node){if(node != nullptr){m_queue.push_back(node);}}

    ~ntree_bfs_iterator() {}
       
    self_type& operator=(const self_type& other) {m_queue = other.m_queue;  return *this;}

    bool operator==(const self_type& other) const{return m_queue == other.m_queue;}
    bool operator!=(const self_type& other) const{return m_queue != other.m_queue;}

    reference operator*() const{return *(m_queue.front());}
    pointer operator->() const{return m_queue.front();}

    self_type& operator++()
    {
        if(m_queue.empty()){return *this;}
        pointer u(m_queue.front());
        m_queue.pop_front();

        for(reference ch : *u){m_queue.push_back(&ch);}
        return *this;
    }

    self_type operator++(int){self_type ret(*this); ++(*this);  return ret;}
};  //class ntree_bfs_iterator


template <typename Node, bool is_const, typename Alloc>
class ntree_leaf_iterator
{
public:
    typedef Node node_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::forward_iterator_tag iterator_category;


    typedef typename tmp::choose<is_const, const Node*, Node*>::type pointer;
    typedef typename tmp::choose<is_const, const Node&, Node&>::type reference;

    typedef ntree_leaf_iterator self_type;

private:
    ntree_dfs_preorder_iterator<Node, is_const, Alloc> m_pre;

public:
    ntree_leaf_iterator() : m_pre() {}
    ntree_leaf_iterator(const self_type& other) : m_pre(other.m_pre) {}
    ntree_leaf_iterator(pointer node) : m_pre(node) 
    {
        if(node != nullptr)
        {
            if(!node->is_leaf())
            {
                this->operator++();
            }
        }
    }
    ~ntree_leaf_iterator() {}

    self_type& operator=(const self_type& other){m_pre = other.m_pre;}
    bool operator==(const self_type& other) const{return m_pre == other.m_pre;}
    bool operator!=(const self_type& other) const{return m_pre != other.m_pre;}

    reference operator*() const{return m_pre.operator*();}
    pointer operator->() const{return m_pre.operator->();}

    self_type& operator++()
    {
        ++m_pre;
        if(m_pre.empty()){return *this;}
        while(!m_pre->is_leaf())
        {
            ++m_pre;
            if(m_pre.empty()){return *this;}
        }
        return *this;
    }

    self_type operator++(int){self_type ret(*this); ++(*this);  return ret;}
};

}   //namespace ttns

#endif  //  TTNS_DATASTRUCTURES_NTREE_ITERATORS_HPP    //

