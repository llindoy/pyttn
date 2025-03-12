#ifndef TTNS_MS_TENSOR_NODE_HPP
#define TTNS_MS_TENSOR_NODE_HPP


#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>
#include "../tree/tree.hpp"
#include "../tree/tree_node.hpp"
#include "../tree/ntree.hpp"

#include <vector>
#include <stdexcept>

#include "ttn_node.hpp"

namespace ttns
{
template <typename T, typename backend>
using multiset_node_data = std::vector<ttn_node_data<T, backend>>;

template <typename T, typename backend> 
std::ostream& operator<<(std::ostream& os, const multiset_node_data<T, backend>& t)
{
    for(size_t i = 0; i < t.size(); ++i)
    {
        os << "set index: " << i << std::endl << t[i] << std::endl;
    }
    return os;
}

}

#include "node_traits/ttn_node_traits.hpp"
#include "node_traits/ms_ttn_node_traits.hpp"
#include "node_traits/tensor_node_traits.hpp"

#include "../orthogonality/decomposition_engine.hpp"
#include "../orthogonality/root_to_leaf_decomposition.hpp"
#include "../orthogonality/leaf_to_root_decomposition.hpp"

namespace ttns
{  


template <typename T, typename backend> 
class tree_node<tree_base<multiset_node_data<T, backend> > >: 
    public tree_node_base<tree_base<multiset_node_data<T, backend> > >
{
    static_assert(std::is_base_of<linalg::backend_base, backend>::value, "The second template argument to the ttn_node object must be a valid backend.");
public:
    using matrix_type = linalg::matrix<T, backend>;
    using value_type = multiset_node_data<T, backend>;
    using tree_type = tree_base<value_type>;
    using base_type = tree_node_base<tree_type>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using node_type = tree_node<tree_base<value_type>>;
    using self_type = node_type;
    using hrank_type = std::vector<size_type>;

    using bond_matrix_type = std::vector<matrix_type>;
    using population_matrix_type = std::vector<linalg::diagonal_matrix<real_type, backend>>;
    using node_helper = ttn_node_helper<multiset_node_data, T, backend>;

    using engine_type = orthogonality::decomposition_engine<T, backend, false>;

    class orthogonality_type
    {
    public:
        using r2l_core = orthogonality::root_to_leaf_decomposition_engine<T, backend>;
        orthogonality_type(){}
        orthogonality_type(const orthogonality_type& o) = default;
        orthogonality_type(orthogonality_type&& o) = default;

        orthogonality_type& operator=(const orthogonality_type& o) = default;
        orthogonality_type& operator=(orthogonality_type&& o) = default;

        bool is_initialised() const{return m_initialised;}

        template <typename tree>
        void init(tree& nodes)
        {
            clear();
            query_sizes(nodes);

            size_type nset = nodes.nset();

            if(m_nthreads < 1){m_nthreads=1;}
            if(m_nthreads > nset){m_nthreads=nset;}

            m_workspace.resize(nset);
            m_U.resize(nset);
            m_R.resize(nset);
            m_S.resize(nset);
            m_ortho_engine.resize(m_nthreads);

            for(size_type i = 0; i < nset; ++i) 
            {
                m_workspace[i].reallocate(m_maxcapacity[i]);
                m_U[i].reallocate(m_maxcapacity[i]);
                m_U[i].resize(1, m_maxsize[i]);

                for(const auto& a : nodes)
                {
                    CALL_AND_HANDLE(r2l_core::resize_r_matrix(a()[i], m_R[i], true), "Failed to resize elements of the r tensor.");
                    m_S[i].resize(m_R[i].shape(0), m_R[i].shape(1));
                }
            }
    
            for(size_type i = 0; i < m_nthreads; ++i)
            {
                try
                {
                    m_ortho_engine[i].template resize<r2l_core>(nodes, m_U, m_R, true);
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to resize the decomposition engine object.");
                }
            }
            m_initialised = true;
        }

        void resize_data(const self_type& a)
        {
            ASSERT(a().size() == m_U.size(), "Cannot resize data for multiset orthogonality type.");
            for(size_type i = 0; i < m_U.size(); ++i)
            {
                m_U[i].resize(a()[i].shape(0), a()[i].shape(1));
                m_workspace[i].resize(a()[i].shape(0), a()[i].shape(1));
            }
        }

        void clear()
        {
            for(size_type i = 0; i < m_ortho_engine.size(); ++i){m_ortho_engine[i].clear();}
            m_ortho_engine.clear();

            for(size_type i = 0; i < m_U.size(); ++i)
            {
                m_U[i].clear();
                m_R[i].clear();
                m_S[i].clear();
                m_workspace[i].clear();
            }
            m_U.clear();
            m_R.clear();
            m_S.clear();
            m_workspace.clear();

            m_maxcapacity.clear();
            m_maxsize.clear();

            m_most_recent_node = 0;
            m_initialised = false;
        }

        template <typename tree>
        void query_sizes(tree& nodes)
        {
            size_type nset = nodes.nset();
            m_maxsize.resize(nset);
            m_maxcapacity.resize(nset);

            for(size_type i = 0; i < nset; ++i)
            {
                std::array<size_type, 3> sizes{{0,0,0}};
                for(const auto& a : nodes){ttn_node_data<T, backend>::query_node_sizes(a()[i], sizes, a.is_leaf());}
                m_maxsize[i] = sizes[0];
                m_maxcapacity[i] = (sizes[1] > sizes[2] ? sizes[1] : sizes[2]);
            }
        }

        orthogonality::truncation_mode& truncation_mode(){return m_truncation_mode;}
        const orthogonality::truncation_mode& truncation_mode() const{return m_truncation_mode;}

        size_type nthreads() const{return m_nthreads;}
        size_type& nthreads() {m_initialised = false; return m_nthreads;}

        bool parallelise() const{return m_nthreads>1;}

        size_type most_recent_node() const{return m_most_recent_node;}
        size_type& most_recent_node(){return m_most_recent_node;}

        population_matrix_type& population_matrix(){return m_S;}
        std::vector<matrix_type>& bond_matrix(){return m_R;}
        std::vector<engine_type>& eng(){return m_ortho_engine;}
        std::vector<matrix_type>& work(){return m_workspace;}
        std::vector<matrix_type>& R(){return m_R;}
        std::vector<matrix_type>& U(){return m_U;}
        population_matrix_type& S(){return S;}
        
        const population_matrix_type& population_matrix() const{return m_S;}
        const std::vector<matrix_type>& bond_matrix() const{return m_R;}
        const std::vector<engine_type>& eng() const{return m_ortho_engine;}
        const std::vector<matrix_type>& work() const{return m_workspace;}
        const std::vector<matrix_type>& R() const{return m_R;}
        const std::vector<matrix_type>& U() const{return m_U;}
        const population_matrix_type& S() const{return m_S;}

        linalg::diagonal_matrix<real_type, backend>& population_matrix(size_type i){return m_S[i];}
        matrix_type& bond_matrix(size_type i){return m_R[i];}
        engine_type& eng(size_type i)
        {       
            ASSERT(i < m_nthreads, "Cannot access engine object index out of bounds.");
            return m_ortho_engine[i];
        }
        matrix_type& work(size_type i){return m_workspace[i];}
        matrix_type& R(size_type i){return m_R[i];}
        matrix_type& U(size_type i){return m_U[i];}
        linalg::diagonal_matrix<real_type, backend>& S(size_type i){return m_S[i];}
        
        const linalg::diagonal_matrix<real_type, backend>& population_matrix(size_type i)const{return m_S[i];}
        const matrix_type& bond_matrix(size_type i) const{return m_R[i];}
        const engine_type& eng(size_type i) const
        {   
            ASSERT(i < m_nthreads, "Cannot access engine object index out of bounds.");
            return m_ortho_engine[i];
        }
        const matrix_type& work(size_type i) const{return m_workspace[i];}
        const matrix_type& R(size_type i) const{return m_R[i];}
        const matrix_type& U(size_type i) const{return m_U[i];}
        const linalg::diagonal_matrix<real_type, backend>& S(size_type i)const{return m_S[i];}
        
    protected:
        std::vector<engine_type> m_ortho_engine;
        std::vector<matrix_type> m_U;
        std::vector<matrix_type> m_R;
        population_matrix_type m_S;
        std::vector<matrix_type> m_workspace;
        std::vector<size_type> m_maxsize;
        std::vector<size_type> m_maxcapacity;
        size_type m_most_recent_node = 0;
        size_type m_nthreads=1;
        bool m_initialised = false;

        orthogonality::truncation_mode m_truncation_mode = orthogonality::truncation_mode::singular_values_truncation;
    };

protected:
    using base_type::m_data;
    using base_type::m_children;
    using base_type::m_parent;

public:
    friend class tree<value_type>;
    friend class tree_base<value_type>;
    friend ms_ttn<T, backend>;
    template <template <typename, typename> class nt, typename U, typename be> friend class ttn_base;

public:
    std::ostream& output_bond_dimensions(std::ostream& os, size_type ind)
    {
        os << "(";
        os << hrank(ind);
        for(size_t i=0; i<this->size(); ++i)
        {
            this->operator[](i).output_bond_dimensions(os, ind);
        }
        os << ")";
        return os;
    }


public:
    tree_node() : base_type(){}

    void zero()
    {
        for(size_type i = 0; i < m_data.size(); ++i)
        {
            m_data[i].fill_zeros();   
        }
    }

    size_type maxsize() const
    {
        size_type ms = 0;
        for(size_type i = 0; i < m_data.size(); ++i)
        {
            if(m_data[i].size() > ms){ms = m_data[i].size();}
        }
        return ms;
    }
    size_type maxcapacity() const
    {
        size_type ms = 0;
        for(size_type i = 0; i < m_data.size(); ++i)
        {
            if(m_data[i].capacity() > ms){ms = m_data[i].capacity();}
        }
        return ms;
    }
    size_type maxhrank(bool use_capacity = false) const
    {
        size_type ms = 0;
        for(size_type i = 0; i < m_data.size(); ++i)
        {
            if(m_data[i].hrank(use_capacity) > ms){ms = m_data[i].hrank(use_capacity);}
        }
        return ms;
    }

    void get_hrank(std::vector<size_type>& res ) const
    {
        res.resize(m_data.size()); 
        for(size_t i = 0; i < m_data.size(); ++i){res[i] = m_data[i].hrank();}
    }

    void get_max_hrank(std::vector<size_type>& res ) const
    {
        res.resize(m_data.size()); 
        for(size_t i = 0; i < m_data.size(); ++i){res[i] = m_data[i].max_hrank();}
    }
    size_type hrank(size_type i) const{return m_data[i].hrank();}
    size_type nmodes() const{return m_children.size();}
    size_type dimen(size_type i) const {return m_data[i].dimen();}
    size_type nset() const {return m_data.size();}
    size_type dim(size_type i, size_type n) const{return m_data[i].dim(n);}
    size_type max_dim(size_type n) const
    {
        size_t mdim = 0;
        for(size_t i = 0; i < nset(); ++i)
        {
            size_t dim = m_data[i].max_dim(n);
            if(mdim > dim){mdim = dim;}
        }
        return mdim;
    }
    const std::vector<size_type>& dims(size_type i) const{return m_data[i].dims();}
    size_type nbonds() const{return this->size() + (this->is_root() ? 0 : 1);}

    size_type buffer_maxcapacity() const
    {
        size_t bs = 0;
        for(size_t i = 0; i < nset(); ++i)
        {
            bs += m_data[i].capacity();
        }
        return bs;
    }

    size_t buffer_size() const
    {
        size_t bs = 0;
        for(size_t i = 0; i < nset(); ++i)
        {
            bs += m_data[i].size();
        }
        return bs;
    }

    const value_type& operator()() const {return m_data;}
    value_type& operator()() {return m_data;}

    const ttn_node_data<T, backend>& operator()(size_type i) const {return m_data[i];}
    ttn_node_data<T, backend>& operator()(size_type i) {return m_data[i];}

    const ttn_node_data<T, backend>& dataview(size_type i) const{return m_data[i];}
    ttn_node_data<T, backend>& dataview(size_type i){return m_data[i];}

#ifdef CEREAL_LIBRARY_FOUND
    friend class serialisation_node_save_wrapper<node_type, size_type>;
    friend class serialisation_node_load_wrapper<node_type, size_type>;
public:
    template <typename archive>
    void save(archive& ar) const
    {
        for
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn_node object.  Error when serialising the base object.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn_node object.  Error when serialising the base object.");
    }
#endif

public:
    template <typename U, typename be>
    bool can_fit_node(const multiset_node_data<U, be>& o) const
    {
        if(o.size() != nset()){return false;}
        for(size_type i = 0; i < nset(); ++i)
        {
            if(!m_data[i].can_fit_node(o[i])){return false;}
        }
        return true;
    }

public:
    real_type norm() const
    {
        real_type _norm = 0;
        for(size_type i = 0; i < this->nset(); ++i)
        {
            auto vec = m_data[i].as_rank_1();
            _norm += linalg::real(linalg::dot_product(linalg::conj(vec), vec));
        }
        return std::sqrt(_norm);
    }

    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, self_type&>::type operator*=(const U& u)
    {
        for(size_type i = 0; i < this->nset(); ++i){m_data[i].as_matrix() *= u;}
        return *this;
    }

    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, self_type&>::type operator/=(const U& u)
    {
        for(size_type i = 0; i < this->nset(); ++i){m_data[i].as_matrix() /= u;}
        return *this;
    }

public:
    template <typename Itype, typename Atype>
    typename std::enable_if<std::is_integral<Itype>::value, void>::type 
    setup_data_from_topology_node(const ntree_node<ntree<Itype, Atype>>& tree_iter, 
                                  const ntree_node<ntree<Itype, Atype>>& capacity_iter, 
                                  size_type ndims,
                                  size_type nset = 1,
                                  bool purification = false)
    {
        m_data.resize(nset);
        for(size_type i = 0; i < nset; ++i)
        {
            m_data[i].setup_data_from_topology_node(tree_iter, capacity_iter, ndims, purification);
        }
    }

    template <typename Itype, typename Atype>
    typename std::enable_if<std::is_integral<Itype>::value, void>::type 
    setup_data_from_topology_node(const ntree_node<ntree<std::vector<Itype>, Atype>>& tree_iter, 
                                  const ntree_node<ntree<std::vector<Itype>, Atype>>& capacity_iter, 
                                  size_type ndims,
                                  size_type nset = 1,
                                  bool purification = false)
    {
        ASSERT(tree_iter.value().size() == nset, "Cannot setup data from topology node the nset variable has not been set correctly.");
        m_data.resize(nset);
        for(size_type i = 0; i < nset; ++i)
        {
            m_data[i].setup_data_from_topology_node(tree_iter, capacity_iter, ndims, i, purification);
        }
    }

public:
    void set_node_identity(){for(auto& data : m_data){data.set_identity();}}
    void set_node_identity(size_type i){m_data[i].set_identity();}
    void set_node_random(linalg::random_engine<backend>& rng){for(size_type i = 0; i < this->nset(); ++i){m_data[i].set_random(rng);}}
    void set_node_random(size_type i, linalg::random_engine<backend>& rng){rng.fill_random(m_data[i]);}

    void set_leaf_node_state(size_type sind, size_type i, linalg::random_engine<backend>& rng, bool random_unoccupied_initialisation=false)
    {
        ASSERT(this->is_leaf(), "Function is only applicable for leaf state nodes.");
        this->m_data[sind].set_node_state(i, rng, random_unoccupied_initialisation);
    }

    template <typename U, typename be> 
    void set_leaf_node_vector(size_type sind, const linalg::vector<U, be>& psi0, linalg::random_engine<backend>& rng)
    {
        ASSERT(this->is_leaf(), "Function is only applicable for leaf state nodes.");
        this->m_data[sind].set_node_vector(psi0, rng);
    }

    void set_leaf_purification(size_type sind, linalg::random_engine<backend>& rng)
    {
        ASSERT(this->is_leaf(), "Function is only applicable for leaf state nodes.");
        this->m_data[sind].set_node_purification(rng);
    }

    void conj()
    {
        for(size_t i = 0; i < this->nset(); ++i)
        {
            this->m_data[i].conj();
        }
    }
public:
    //Functions for handling subsections of the movement of the orthogonality centre up or down
    void decompose_down(orthogonality_type& orth, size_type mode, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false){CALL_AND_RETHROW(node_helper::decompose_down(*this, orth, mode, tol, nchi, save_svd));}
    void decompose_up(orthogonality_type& orth, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false){CALL_AND_RETHROW(node_helper::decompose_up(*this, orth, tol, nchi, save_svd));}

    //Function for applying results of orthogonality decomposition to a node
    void apply_to_node(orthogonality_type& orth){CALL_AND_RETHROW(node_helper::apply_to_node(*this, orth));}
  
    //Functions for applying bond matrix.  Here we implement four functions that allow for us to apply bond matrices above this node
    //either to this node or its parent, or apply bond matrices below this node either to this node or its child
    void apply_bond_matrix_to_parent(orthogonality_type& orth){CALL_AND_RETHROW(node_helper::apply_bond_matrix_to_parent(*this, orth));}
    void apply_bond_matrix_from_parent(orthogonality_type& orth){CALL_AND_RETHROW(node_helper::apply_bond_matrix_from_parent(*this, orth));}
    void apply_bond_matrix_to_child(orthogonality_type& orth, size_type mode){CALL_AND_RETHROW(node_helper::apply_bond_matrix_to_child(*this, orth, mode));}
    void apply_bond_matrix_from_child(orthogonality_type& orth, size_type mode){CALL_AND_RETHROW(node_helper::apply_bond_matrix_from_child(*this, orth, mode));}

    //Functions for handling the movement of the orthogonality centre up or down
    void shift_orthogonality_down(orthogonality_type& orth, size_type mode, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false){CALL_AND_RETHROW(node_helper::shift_orthogonality_down(*this, orth, mode, tol, nchi, save_svd));}
    void shift_orthogonality_up(orthogonality_type& orth, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false){CALL_AND_RETHROW(node_helper::shift_orthogonality_up(*this, orth, tol, nchi, save_svd));}
};

}   //namespace ttns


#endif  //TTNS_TENSOR_NODE_HPP//

