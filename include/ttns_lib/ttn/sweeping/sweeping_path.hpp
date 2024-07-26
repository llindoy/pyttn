#ifndef TTNS_LIB_SWEEPING_ALGORITHMS_SWEEPING_PATH_HPP
#define TTNS_LIB_SWEEPING_ALGORITHMS_SWEEPING_PATH_HPP

#include <vector>
#include <common/exception_handling.hpp>

namespace ttns
{

namespace sweeping
{
//a class for representing a general traversal order through a tree structure
class traversal_path
{
public:
    using size_type = std::size_t;

    using iterator = typename std::vector<size_type>::iterator;
    using const_iterator = typename std::vector<size_type>::const_iterator;
    using reverse_iterator = typename std::vector<size_type>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<size_type>::const_reverse_iterator;

protected:
    std::vector<size_type> m_traversal_order;       //the order that we traverse the path
    std::vector<size_type> m_times_visited;         //the number of times that this tree has been traversed in the current path
    std::vector<size_type> m_visits;                //the total number of traversals we will make through the tree

public:
    template <typename V>
    static inline void initialise_euler_tour(const tree<V>& A, traversal_path& op)
    {
        try
        {
            std::vector<size_type> times_visited(A.size()); std::fill(std::begin(times_visited), std::end(times_visited), 0);
            //resize the traversal order array
            size_type ntraversal_sites = 0;
            for(const auto& a : A)
            {
                ntraversal_sites += a.is_leaf() ? 2 : (1 + a.size());
            }
            std::vector<size_type> traversal_order(ntraversal_sites);


            //now initialise the traversal order array
            const auto* curr_node = &A.root();
            for(size_type i=0; i<ntraversal_sites; ++i)
            {
                size_type curr_node_id = curr_node->id();
                traversal_order[i] = curr_node_id;

                size_type n_times_visited = times_visited[curr_node_id];

                if(n_times_visited == 0 && curr_node->is_leaf()){}
                else if(n_times_visited < curr_node->size())
                {
                    CALL_AND_HANDLE
                    (
                        curr_node = curr_node->child_pointer(times_visited[curr_node_id]), 
                        "Failed to access a child of a node when constructing the traversal order array."
                    );
                }
                else if(!curr_node->is_root())
                {
                    CALL_AND_HANDLE
                    (
                        curr_node = curr_node->parent_pointer(), 
                        "Failed to access parent of node when constructing the traversal order array."
                    );
                }
                else if(i+1 != ntraversal_sites)
                {
                    std::cerr << i << " " << ntraversal_sites << std::endl;
                    RAISE_EXCEPTION("Critical Error: This condition should never be meet");
                }

                ++times_visited[curr_node_id];
            }

            op.initialise(traversal_order);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize tdvp_tree_traversal object.");
        }
    }

public:
    traversal_path(){}

    traversal_path(const std::vector<size_type>& traversal_order)
    {
        initialise(traversal_order);
    }

    traversal_path(const traversal_path& o) = default;
    traversal_path(traversal_path&& o) = default;

    traversal_path& operator=(const traversal_path& o) = default;
    traversal_path& operator=(traversal_path&& o) = default;
    traversal_path& operator=(const std::vector<size_type>& o)
    {
        initialise(o);
        return *this;
    }

    void initialise(const std::vector<size_type>& path)
    {
        //first we set the traversal order array.  This stores the order in which we visit the nodes
        m_traversal_order = path;

        //get the maximum index that is reached in this path
        size_type nterms = 0;
        for(const auto& ind : m_traversal_order)
        {
            if(ind > nterms){nterms = ind;}
        }

        m_times_visited.resize(nterms+1); std::fill(m_times_visited.begin(), m_times_visited.end(), 0);
        m_visits.resize(nterms+1); std::fill(m_times_visited.begin(), m_times_visited.end(), 0);

        for(const auto& ind : m_traversal_order){++m_visits[ind];}
    }

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_traversal_order.clear(), "Failed to clear traversal order array.");
            CALL_AND_HANDLE(m_times_visited.clear(), "Failed to clear times visited array.");
            CALL_AND_HANDLE(m_visits.clear(), "Failed to clear times visited array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear tdvp_tree_traversal object.");
        }
    }

    void reset_visits(){CALL_AND_HANDLE(std::fill(m_times_visited.begin(), m_times_visited.end(), 0), "Failed to reset the number of times each node was visited.");}
    void visit(size_type i){++m_times_visited[i];}
    size_type times_visited(size_type i){return m_times_visited[i];}

    bool first_visit(size_type i ) const{return (m_times_visited[i] == 1);}
    bool last_visit(size_type i) const{return (m_times_visited[i] == m_visits[i]);}
public:
    //iterator functions
    iterator begin() {  return iterator(m_traversal_order.begin());  }
    iterator end() {  return iterator(m_traversal_order.end());  }
    const_iterator begin() const {  return const_iterator(m_traversal_order.begin());  }
    const_iterator end() const {  return const_iterator(m_traversal_order.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_traversal_order.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_traversal_order.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_traversal_order.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_traversal_order.rend());  }
};

}   //namespace sweeping


}

#endif

