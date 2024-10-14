///
/// @file ntree_builder.hpp
/// @author Lachlan Lindoy
/// 
/// @brief File containing classes for building ntree object
/// 
///
#ifndef NTREE_BUILDER_HPP
#define NTREE_BUILDER_HPP

#include <common/exception_handling.hpp>

#include "ntree.hpp"
#include <vector>

namespace ttns
{

template <typename T>
class ntree_builder
{
public:
    using tree_type = ntree<T>;
    using node_type = typename tree_type::node_type;
    using size_type = typename tree_type::size_type;
    using leaf_index = std::vector<std::vector<size_type>>;


protected:
    static size_type get_nlevels_for_tree(size_type nchild, size_type nbranch)
    {
        return 1;
    }

public:
    /*  
    class linear
    {
    public:
        linear() : m_nlevels(1), m_ntop(1); m_nbottom(1){}
        linear(size_type top, size_type bottom, size_type nl) : m_nlevels(nl), m_ntop(top), m_nbottom(bottom) {}
        linear(size_type top, size_type bottom, size_type nchild, size_type nbranch) : m_ntop(top), m_nbottom(bottom)
        {   
            set_nlevels_for_tree(nchild, nbranch);
        }

        void set_nlevels_for_tree(size_type nchild, size_type nbranch)
        {
            m_nlevels = get_nlevels_for_tree(nchild, nbranch);
        }

        size_type operator()(size_t l) const
        {
              
        }

    protected:
        size_type m_nlevels;
        size_type m_ntop;
        size_type m_nbottom;
    };

    class logarithmic
    {
    public:
        logarithmic() : m_nlevels(1), m_ntop(1); m_nbottom(1){}
        logarithmic(size_type top, size_type bottom, size_type nl) : m_nlevels(nl), m_ntop(top), m_nbottom(bottom) {}
        logarithmic(size_type top, size_type bottom, size_type nchild, size_type nbranch) : m_ntop(top), m_nbottom(bottom)
        {   
            set_nlevels_for_tree(nchild, nbranch);
        }

        void set_nlevels_for_tree(size_type nchild, size_type nbranch)
        {
            m_nlevels = get_nlevels_for_tree(nchild, nbranch);
        }

        size_type operator()(size_t l) const
        {
              
        }

    protected:
        size_type m_nlevels;
        size_type m_ntop;
        size_type m_nbottom;
    };*/

public:
    /*
     *  Functions for constructing balanced N-ary trees with values specified either by a function of the level or as a constant value.
     *  These functions also return a vector containing vectors indexing the leaf indice that allows for easy addition of nodes to the leaves
     *  of this tree.
     */
    template <typename Func>
    static tree_type balanced_tree(size_type Nleaves, size_type degree, Func&& fl, std::vector<std::vector<size_type>>& linds)
    {
        ASSERT(degree > 1, "Failed to construct a balanced tree that does not branch.");
        ASSERT(Nleaves > 0, "Cannot construct a tree with zero leaves.");
        if(linds.size() != Nleaves){linds.resize(Nleaves);}
    
        //reserve the storage for each of the index arrays.  The maximum length of any of the index arrays is ceil(log_degree(Nleaves))
        double ceil_log = std::ceil(std::log(Nleaves)/std::log(degree));
        size_type depth = static_cast<size_type>(ceil_log);
        for(size_type i=0; i < Nleaves; ++i)
        {
            linds.reserve(depth+1);
        }

        tree_type tree; tree.insert(T(1));
        if(Nleaves != 1)
        {
            balanced_subtree(tree(), Nleaves, degree, std::forward<Func>(fl), linds, false);
        }
        return tree;
    }

    /*
     *  Functions for constructing either a balanced degree-ary subtree or degenerate subtree with a root given by root that is specified in some already defined treein a tree that has already 
     */
    template <typename Func>
    static void balanced_subtree(node_type& root, size_type Nleaves, size_type degree, Func&& fl, std::vector<std::vector<size_type>>& linds, bool allocate = true)
    {
        ASSERT(degree > 1, "Failed to append balanced subtree.  This routine does not work with trees that do not branch.")
        ASSERT(Nleaves > 0, "Failed to append balanced subtree. Cannot create tree with no leafs.");

        if(allocate)
        {
            if(linds.size() != Nleaves){linds.resize(Nleaves);}
            double ceil_log = std::ceil(std::log(Nleaves)/std::log(degree));
            size_type depth = static_cast<size_type>(ceil_log);
            for(size_type i=0; i < Nleaves; ++i)
            {
                linds.reserve(depth);
            }
        }

        if(Nleaves < degree)
        {
            for(size_t i=0; i<Nleaves; ++i)
            {
                linds[i].resize(1);
                linds[i][0] = root.size();
                root.insert(evaluate_value(fl,1));
            }
        }
        else
        {
            size_t r = Nleaves%degree;
            size_type count = 0;
            for(size_t i=0; i<degree; ++i)
            {
                size_t Nchild = Nleaves/degree + (i < r ? 1 : 0);
                for(size_type j=0; j<Nchild; ++j)
                {
                    linds[count+j].push_back(root.size());
                }
                root.insert(evaluate_value(fl,1));
                balanced_subtree_internal(degree, root.back(), Nchild, std::forward<Func>(fl), linds, 2, count);
                count+=Nchild;
            }
        }
    }
protected:
    template <typename Func>
    static void balanced_subtree_internal(size_t degree, node_type& node, size_t nadd, Func&& fl, std::vector<std::vector<size_type>>& linds,  size_type level, size_type count)
    {
        if(nadd < degree)
        {
            if(nadd == 1)
            {
                return ;
            }
            else
            {
                for(size_t i=0; i < nadd; ++i)  
                {
                    linds[count+i].push_back(node.size());
                    node.insert(evaluate_value(fl,level));
                }
                return;
            }
        }
        else
        {
            size_t r = nadd%degree;
            for(size_t i = 0; i < degree; ++i)
            {
                size_t Nchild = nadd/degree + (i < r ? 1 : 0);
                for(size_type j=0; j<Nchild; ++j)
                {
                    linds[count+j].push_back(node.size());
                }
                node.insert(evaluate_value(fl,level));
                balanced_subtree_internal(degree, node.back(), Nchild, std::forward<Func>(fl), linds, level+1, count);
                count+=Nchild;
            }
            return ;
        }       
    }

protected:
    template <typename F>
    static inline typename std::enable_if<std::is_convertible<F,T>::value, T>::type evaluate_value(F t, size_type/* l */)
    {
        return t;
    }

    template <typename F>
    static inline typename std::enable_if<!std::is_convertible<F,T>::value, T>::type evaluate_value(F t, size_type l)
    {
        return t(l);
    }


    static inline T evaluate_value(const std::vector<T>& t, size_type l)
    {
        return t[l];
    }

public:
    /*
     *  Functions fo constructing degenerate trees with values in the tree specified by a function of the level or as a constant value.  
     */
    template <typename Func>
    static tree_type degenerate_tree(size_type Nnodes, Func&& fl, std::vector<std::vector<size_type>>& linds)
    {
        ASSERT(Nnodes > 0, "Cannot construct a tree with zero leaves.");

        tree_type tree; tree.insert(T(1));
        linds.resize(Nnodes);
        for(size_type i = 0; i < Nnodes; ++i)
        {
            linds[i].resize(i);
            for(size_type j=0; j < linds[i].size(); ++j)
            {
                linds[i][j] = 0;
            }
        }
        if(Nnodes != 1)
        {        
            degenerate_subtree(tree(), Nnodes-1, std::forward<Func>(fl), linds, false);
        }
        return tree;
    }

    template <typename Func>
    static void degenerate_subtree(node_type& node, size_type Nnodes, Func&& fl, std::vector<std::vector<size_type>>& linds, bool allocate = true)
    {
        ASSERT(Nnodes > 0, "Failed to append degenerate sybtree. Cannot create tree with no leafs.");
        if(allocate)
        {
            linds.resize(Nnodes);
            for(size_type i = 0; i < Nnodes; ++i)
            {
                linds[i].resize(i+1);
                linds[i][0] = node.size();
                for(size_type j=1; j < linds[i].size(); ++j)
                {
                    linds[i][j] = 0;
                }
            }
        }

        degenerate_subtree_internal(node, Nnodes, std::forward<Func>(fl), 0);
    }

protected:
    template <typename Func>
    static void degenerate_subtree_internal(node_type& node, size_type Nnodes, Func&& fl, size_type level)
    {
        if(level+1 < Nnodes)
        {
            node.insert(evaluate_value(fl, level));
            degenerate_subtree_internal(node.back(), Nnodes, std::forward<Func>(fl), level+1);
        }
        else
        {
            node.insert(evaluate_value(fl, level));
        }
    }

protected:
    static void collapse_trivial(tree_type& tree, node_type& node, size_type index)
    {
        //we only attempt a trivial node collapse if the current node is not a leaf.  The collapse function will retain the smaller of the two bonds
        if(!node.is_leaf())
        {
            //if the node only has one child but the child is not a leaf.  Then we want to collapse this node with its child
            if(node.size() == 1 && !(node[0].is_leaf()))
            {
                node_type* curr_node = &node;
                node_type* child_node = node.m_children[0];
                child_node->decrement_level();

                //set the child nodes parent to the current nodes parent
                bool is_root = node.is_root();
                child_node->m_parent = curr_node->m_parent;

                //set the child node data to the minimum of the two nodes data
                if(curr_node->m_data < child_node->m_data){child_node->m_data = curr_node->m_data;}

                //decrement the subtree size associated with the ancestor nodes of the current node
                node_type* q = curr_node;
                while(!q->is_root())
                {
                    q = q->m_parent;
                    q->m_size -= 1;
                }

                if(!is_root){curr_node->m_parent->m_children[index] = child_node;}

                
                curr_node->m_children[0] = nullptr;
                curr_node->m_parent = nullptr;

                curr_node->m_children.clear();
                tree.destroy_node(curr_node);
                curr_node = nullptr;

                if(is_root){tree.m_root = child_node;}

                collapse_trivial(tree, *child_node, index);

                child_node = nullptr;
            }
            else
            {
                for(size_t i = 0; i < node.size(); ++i)
                {
                    collapse_trivial(tree, node[i], i);
                }
            }
        }
    }

public:
    //iterate through the tree and anytime an internal bond matrix is found collapse this into the neighbouring nodes
    static void collapse_bond_matrices(tree_type& tree)
    {
        if(tree.size() < 2){return;}

        //now iterate through the tree and see if there are any modes that have a single child and that are not the parents of leaf nodes.
        //In this case we collapse the tree
        collapse_trivial(tree, tree.root(), 0);

        //first check if the root node is a bond tensor and if so contract it into its first child
        if(tree.root().size() == 2)
        {
            //get the current root 
            node_type* root = &tree.root();
            node_type* new_root;
            //if either of the two subtrees are greater than two we do the merging.  If this isn't the case we just skip this step
            if(tree.root()[0].subtree_size() > 2 || tree.root()[1].subtree_size() > 2)
            {
                if(tree.root()[0].subtree_size() >= tree.root()[1].subtree_size())
                {
                    //and the node that will be the new_root
                    new_root = root->m_children[0];
                    new_root->decrement_level();

                    //set the new root to the root
                    new_root->m_parent = nullptr;
                    new_root->m_data = 1;

                    new_root->m_size = root->m_size - 1;
                    new_root->m_nleaves = root->m_nleaves;

                    //and insert the current roots children other child into it
                    new_root->m_children.push_back(root->m_children[1]);
                    new_root->m_children.back()->m_parent = new_root;
                }
                else
                {
                    //and the node that will be the new_root
                    new_root = root->m_children[1];
                    new_root->decrement_level();

                    //set the new root to the root
                    new_root->m_parent = nullptr;
                    new_root->m_data = 1;

                    new_root->m_size = root->m_size - 1;
                    new_root->m_nleaves = root->m_nleaves;

                    //and insert the current roots children other child into it
                    new_root->m_children.insert(new_root->m_children.begin(), root->m_children[0]);
                    new_root->m_children.front()->m_parent = new_root;
                }
            
                //set the current nodes
                for(size_t i = 0; i < tree.root().size(); ++i)
                {
                    root->m_children[i] = nullptr;
                }

                root->m_children.clear();
                tree.destroy_node(root);
                root = nullptr;

                tree.m_root = new_root;
                new_root = nullptr;
            }
        }
    }

    //iterate through the tree and ensure that the bond dimension is not larger than the number of states required to represent the state
    //Currently this only performs the operation up the tree.  
    //TODO: Add an iteration down the tree to ensure that the number of single hole functions is bounded by the number of states that can
    //be represented
    static void sanitise_bond_dimensions(tree_type& tree)
    {
        if(tree.size() < 2){return;}
        //we need to correctly implement the post_order_iterator here
        for(typename tree_type::post_iterator tree_iter = tree.post_begin(); tree_iter != tree.post_end(); ++tree_iter)
        {
            if(!(tree_iter->is_leaf()))
            {
                T size2 = 1;
                //iterate over all of the children of the node and calculate the product of their values.  If the 
                //product of their values is less than the value stored at the current node we set the value at the 
                //current node to their product.
                for(auto& topology_child : *tree_iter)
                {
                    size2 *= topology_child.value();
                }
                if(size2 < tree_iter->value()){tree_iter->value() = size2;}
            }
        }
    }

protected:
    static void insert_local_nodes(node_type& node)
    {
        //for the current node determine how many of its children are not leaves 
        //if the node has a child which is a leaf.  But this is not its only child.
        //Then we insert a new node below that child.

        if(!node.is_leaf())
        {
            size_type nchildren = 0;
            size_type nleaves = 0;
            for(auto& child : node)
            {
                if(!child.is_leaf()){++nchildren;}
                else{++nleaves;}
            }

            //if the children of this node are all leaf nodes.
            if(nchildren == 0)
            {
                //if the node has 1 or fewer leaf child then we are done.  If it has more than one childn
                //the exterior node has too many children and we will pad each of them with a new mode
                if(nleaves > 1)
                {
                    for(auto& child : node)
                    {
                        child.insert(child());
                    }
                }
            }
            //if the node has a child that is not the leaf node
            else
            {
                //then iterate over the children of this node
                for(auto& child : node)
                {
                    //and if the child is a leaf node.  We insert a new child into the node
                    if(child.is_leaf())
                    {
                        child.insert(child());
                    }
                    else
                    {
                        insert_local_nodes(child);
                    }
                }
            }
        }
    }


public:
    static void insert_basis_nodes(tree_type& tree)
    {
        if(tree.size() < 2){return;}
        insert_local_nodes(tree.root());
    }

    static void sanitise_tree(tree_type& tree, bool remove_bond_matrices = true)
    {
        if(tree.size() < 2){return;}

        insert_basis_nodes(tree);
        if(remove_bond_matrices)
        {
            collapse_bond_matrices(tree);
        }
        sanitise_bond_dimensions(tree);
    }

public:
    template <typename Func>
    static tree_type htucker_tree(const std::vector<T>& Hb, size_type degree, Func&& fl, bool include_local_basis_transform = true)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        tree_type ret;
        CALL_AND_HANDLE(ret = balanced_tree(Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced tree.");
        
        for(size_type i = 0; i < linds.size(); ++i)
        {
            if(include_local_basis_transform)
            {
                ret.at(linds[i]).insert(Hb[i]);
            }
            else
            {
                ret.at(linds[i])() = Hb[i];
            }
        }
        return ret;
    }

    template <typename Func>
    static void htucker_subtree(node_type& root, const std::vector<T>& Hb, size_type degree, Func&& fl, bool include_local_basis_transform = true)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        CALL_AND_HANDLE(balanced_subtree(root, Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced sub tree.");
        for(size_type i = 0; i < linds.size(); ++i)
        {
            if(include_local_basis_transform)
            {
                root.at(linds[i]).insert(Hb[i]);
            }
            else
            {
                root.at(linds[i])() = Hb[i];
            }
        }
    }

    template <typename Func>
    static tree_type htucker_tree(const std::vector<T>& Hb, size_type degree, Func&& fl, const std::vector<T>& lHd)
    {
        size_type Nleaves = Hb.size();
        ASSERT(lHd.size() == Nleaves, "Failed to add in local hilbert space dimension nodes.  Invalid size.");
        std::vector<std::vector<size_type>> linds(Nleaves);
        tree_type ret;
        CALL_AND_HANDLE(ret = balanced_tree(Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced tree.");
        
        for(size_type i = 0; i < linds.size(); ++i)
        {
            ret.at(linds[i])() = lHd[i];
            ret.at(linds[i]).insert(Hb[i]);
        }
        return ret;
    }

    template <typename Func>
    static void htucker_subtree(node_type& root, const std::vector<T>& Hb, size_type degree, Func&& fl, const std::vector<T>& lHd)
    {
        size_type Nleaves = Hb.size();
        ASSERT(lHd.size() == Nleaves, "Failed to add in local hilbert space dimension nodes.  Invalid size.");
        std::vector<std::vector<size_type>> linds(Nleaves);
        CALL_AND_HANDLE(balanced_subtree(root, Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced sub tree.");
        for(size_type i = 0; i < linds.size(); ++i)
        {
            root.at(linds[i])() = lHd[i];
            root.at(linds[i]).insert(Hb[i]);
        }
    }

    template <typename Func>
    static tree_type htucker_tree(const std::vector<T>& Hb, size_type degree, Func&& fl, T lHd)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        tree_type ret;
        CALL_AND_HANDLE(ret = balanced_tree(Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced tree.");
        
        for(size_type i = 0; i < linds.size(); ++i)
        {
            ret.at(linds[i])() = lHd;
            ret.at(linds[i]).insert(Hb[i]);
        }
        return ret;
    }

    template <typename Func>
    static void htucker_subtree(node_type& root, const std::vector<T>& Hb, size_type degree, Func&& fl, T lHd)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        CALL_AND_HANDLE(balanced_subtree(root, Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced sub tree.");
        for(size_type i = 0; i < linds.size(); ++i)
        {
            root.at(linds[i])() = lHd;
            root.at(linds[i]).insert(Hb[i]);
        }
    }

    template <typename ... Args>
    static void mlmctdh_tree(Args&& ... args)
    {
        CALL_AND_RETHROW(htucker_tree(std::forward<Args>(args)...));
    }

    template <typename ... Args>
    static void mlmctdh_subtree(Args&& ... args)
    {
        CALL_AND_RETHROW(htucker_subtree(std::forward<Args>(args)...));
    }

    template <typename Func>
    static tree_type mps_tree(const std::vector<T>& Hb, Func&& fl)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        tree_type ret;
        CALL_AND_HANDLE(ret = degenerate_tree(Nleaves, std::forward<Func>(fl), linds), "Failed to build balanced tree.");
        
        for(size_type i = 0; i < linds.size(); ++i)
        {
            size_t ni = linds.size()-(i+1);
            ret.at(linds[ni]).insert_front(Hb[ni]);
        }
        return ret;
    }

    template <typename Func, typename Func2>
    static tree_type mps_tree(const std::vector<T>& Hb, Func&& fl, Func2&& f2)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        tree_type ret;
        CALL_AND_HANDLE(ret = degenerate_tree(Nleaves, std::forward<Func>(fl), linds), "Failed to build balanced tree.");
        
        for(size_type i = 0; i < linds.size(); ++i)
        {
            size_t ni = linds.size()-(i+1);
            size_type ind  = ret.at(linds[ni]).insert_front(evaluate_value(f2, ni));
            ret.at(linds[ni])[ind].insert(Hb[ni]);
        }
        return ret;
    }

    template <typename Func>
    static void mps_subtree(node_type& root, const std::vector<T>& Hb, Func&& fl)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        CALL_AND_HANDLE(degenerate_subtree(root, Nleaves, std::forward<Func>(fl), linds), "Failed to build balanced sub tree.");
        for(size_type i = 0; i < linds.size(); ++i)
        {
            size_t ni = linds.size()-(i+1);
            root.at(linds[ni]).insert_front(Hb[ni]);
        }
    }

    template <typename Func, typename Func2>
    static void mps_subtree(node_type& root, const std::vector<T>& Hb, Func&& fl, Func2&& f2)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        CALL_AND_HANDLE(degenerate_subtree(root, Nleaves, std::forward<Func>(fl), linds), "Failed to build balanced sub tree.");
        for(size_type i = 0; i < linds.size(); ++i)
        {
            size_t ni = linds.size()-(i+1);
            size_type ind  = root.at(linds[ni]).insert_front(evaluate_value(f2, ni));
            root.at(linds[ni])[ind].insert(Hb[ni]);
        }
    }

};

}   //namespace ttns

#endif

