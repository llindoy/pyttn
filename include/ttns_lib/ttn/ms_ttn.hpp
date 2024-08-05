#ifndef TTNS_MS_TTN_HPP
#define TTNS_MS_TTN_HPP

#include <random>
#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

#include "ttn_nodes/ms_ttn_node.hpp"
#include "ttn_nodes/ttn_node.hpp"

#include "ttnbase.hpp"

namespace ttns
{
template <typename T, typename backend>
using ms_ttn_node = typename tree<multiset_node_data<T, backend> >::node_type;

template <typename T, typename backend = blas_backend>
class ms_ttn : public ttn_base<multiset_node_data, T, backend> 
{
public:
    using base_type = ttn_base<multiset_node_data, T, backend>;
    using real_type = typename base_type::real_type;
    using matrix_type = typename base_type::matrix_type;

    using value_type = typename base_type::value_type;
    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using size_type = typename backend::size_type;

    using node_reference = typename base_type::node_reference;
    using const_node_reference = typename base_type::const_node_reference;

    using tree_type = typename base_type::tree_type;
    using tree_reference = typename base_type::tree_reference;
    using const_tree_reference = typename base_type::const_tree_reference;

    using node_type = typename base_type::node_type;
    using ancestor_index = typename base_type::ancestor_index;

    using bond_matrix_type = typename node_type::bond_matrix_type;

    using slice_type = multiset_ttn_slice<T, backend, false>;
    using const_slice_type = multiset_ttn_slice<T, backend, true>;

    template <typename U, typename be>
    friend class ttn;
private:
    //provide access to base class operators
    using base_type::m_nodes;
    using base_type::m_nleaves;
    using base_type::_rng;
    using base_type::m_orthog;
    using base_type::m_dim_sizes;
    using base_type::m_leaf_indices;
    using base_type::m_has_orthogonality_centre;

public:
    using base_type::size;

public:
    ms_ttn() : base_type(){}

    ms_ttn(const ms_ttn& other) try : base_type(other) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Multiset_TTN object.");
    }

    template <typename U, typename = typename std::enable_if<not std::is_same<T, U>::value, void>::type> 
    ms_ttn(const ms_ttn<U, backend>& other) try : base_type(other){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Multiset_TTN object.");
    }

    template <typename INTEGER, typename Alloc>
    ms_ttn(const ntree<INTEGER, Alloc>& topology, size_type nset, bool purification = false) try : base_type(topology, nset, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Multiset_TTN object.");
    }

    template <typename INTEGER, typename Alloc>
    ms_ttn(const ntree<INTEGER, Alloc>& topology, const ntree<INTEGER, Alloc>& capacity, size_type nset, bool purification = false)try : base_type(topology, capacity, nset, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Multiset_TTN object.");
    }

    ms_ttn(const std::string& _topology, size_type nset, bool purification = false) try : base_type(_topology, nset, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Multiset_TTN object.");
    }

    ms_ttn(const std::string& _topology, const std::string& _capacity, size_type nset, bool purification = false) try : base_type(_topology, _capacity, nset, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Multiset_TTN object.");
    }

public:
    ms_ttn& operator=(const ms_ttn& other) = default;
    template <typename U, typename be>
    ms_ttn& operator=(const ms_ttn<U, be>& other){CALL_AND_RETHROW(return base_type::operator=(other));}

    template <typename U, typename be>
    ms_ttn& assign_set(size_type set, const ttn<U, be>& other)
    {
        ASSERT(set < this->nset(), "Cannot set ms_ttn set, index out of bounds.");
        ASSERT(has_same_structure(*this, other), "The input hiearchical tucker tensors do not both have the same topology as the matrix_element object.");
        for(auto z : common::zip(m_nodes, other))
        {
            std::get<0>(z)()[set] = std::get<1>(z)();
        }
        return *this;
    }

public:
    template <typename int_type> 
    void set_state(size_type sind, const std::vector<int_type>& si){CALL_AND_RETHROW(this->_set_state(si, sind));}

    template <typename U, typename be> 
    void set_product(size_type sind, const std::vector<linalg::vector<U, be> >& ps){CALL_AND_RETHROW(this->_set_product(ps, sind));}

    template <typename Rvec> 
    void sample_product_state(size_type sind, std::vector<size_t>& state, const std::vector<Rvec>& relval){CALL_AND_RETHROW(this->_sample_product_state(state, relval, sind));}

    template <typename U, typename int_type> 
    void set_state(const std::vector<U>& coeff, const std::vector<std::vector<int_type>>& si)
    {
        ASSERT(coeff.size() == si.size() && coeff.size() == this->nset(), "Cannot set ttnbase to specified state.  Input arrays are not the correct size.");

        for(size_type set_index = 0; set_index < this->nset(); ++set_index)
        {
            ASSERT(si[set_index].size() == this->nmodes(), "Cannot set ttnbase to specified state.  The state does not have the required numbers of modes.");

            for(size_type i = 0; i < this->nmodes(); ++i)
            {
                ASSERT(static_cast<size_type>(si[set_index][i]) < m_dim_sizes[i], "Cannot set state, state index out of bounds.");
            }
        }
        
        //now zero the state
        this->zero();

        for(size_type set_index = 0; set_index < this->nset(); ++set_index)
        {
            //set each of the states
            this->initialise_logical_tensors_product_state(set_index);

            for(size_type i = 0; i < this->nmodes(); ++i)
            {
                CALL_AND_HANDLE(m_nodes[m_leaf_indices[i]].set_leaf_node_state(set_index, si[set_index][i], _rng), "Failed to set state.");
            }

            //and scale them by the coeff array
            m_nodes[0]()[set_index] *= coeff[set_index];
        }

        //now enforce that the orthogonality centre is at the root
        this->force_set_orthogonality_centre(0);
    }

    template <typename V, typename U, typename be> 
    void set_product(const std::vector<U>& coeff, const std::vector<linalg::vector<V, be> >& ps)
    {
        ASSERT(coeff.size() == ps.size() && coeff.size() == this->nset(), "Cannot set ttnbase to specified state.  Input arrays are not the correct size.");

        for(size_type set_index = 0; set_index < this->nset(); ++set_index)
        {
            ASSERT(ps[set_index].size() == this->nmodes(), "Cannot set ttnbase to specified state.  The state does not have the required numbers of modes.");
        }        

        //now zero the state
        this->zero();

        for(size_type set_index = 0; set_index < this->nset(); ++set_index)
        {
            //set each of the states
            this->initialise_logical_tensors_product_state(set_index);

            for(size_type i = 0; i < this->nmodes(); ++i)
            {
                m_nodes[m_leaf_indices[i]].set_leaf_node_vector(set_index, ps[set_index][i], _rng );
            }

            //and scale them by the coeff array
            m_nodes[0]()[set_index] *= coeff[set_index];
        }

        //now enforce that the orthogonality centre is at the root
        this->force_set_orthogonality_centre(0);
    }

    void set_purification()
    {
        ASSERT(this->is_purification(), "Set state requires a purification state.");
        this->zero();
        size_type ns = static_cast<size_type>(std::sqrt(this->nset()));
        ASSERT(ns*ns == this->nset(), "Cannot set purification state.  The set dimension is not a perfect square and therefore cannot represent composite system, ancilla states.");

        //now set up the purification state.
        for(size_type i = 0; i < ns; ++i)
        {
            size_type j = (i+1)%ns;
            size_type set_index = i*ns+j;

            //set each of the states
            this->initialise_logical_tensors_product_state(set_index);

            for(size_type mode = 0; mode < this->nmodes(); ++mode)
            {
                m_nodes[m_leaf_indices[mode]].set_leaf_purification(set_index, _rng);
            }

            //and scale them by the coeff array
            m_nodes[0]()[set_index] *= 1.0/std::sqrt(ns);
        }
    }

public:
    slice_type slice(size_type i)
    {
        ASSERT(i < this->nset(), "Cannot access ms_ttn slice index out of bounds.");
        return slice_type(*this, i);
    }
    const_slice_type slice(size_type i) const
    {
        ASSERT(i < this->nset(), "Cannot access ms_ttn slice index out of bounds.");
        return const_slice_type(*this, i);
    }

public:
    size_type nthreads() const{return m_orthog.nthreads();}
    size_type& nthreads() {return m_orthog.nthreads();}

    real_type bond_entropy(size_t bond_index)
    {
/*  
        try
        {
            if(!m_orthog.is_initialised()){m_orthog.init(*this, m_maxsize, m_maxcapacity);}

            ASSERT(this->has_orthogonality_centre(), "The orthogonality centre must be specified in order to allow for it to be shifted.");
            ASSERT(bond_index < m_nodes[m_orthogonality_centre].nbonds(), "Failed to shift orthogonality centre along bond.  Bond index out of bounds.");

            bool bond_shifted = false;
            //if we aren't at the root node - then we first check whether the bond_index is 0 in which case we should shift the 
            if(!m_nodes[m_orthogonality_centre].is_root())
            {
                //if the bond index is zero then we shift the orthogonality centre up the tree
                if(bond_index == 0)
                {
                    //perform the svd of this node
                    CALL_AND_HANDLE(l2r_core::evaluate(m_ortho_engine, m_nodes[m_orthogonality_centre]), "Failed to evaluate the leaf_to_root_decomposition for a given node.");
                    bond_shifted = true;
                }
                else
                {
                    --bond_index;
                }
            }
            if(!bond_shifted)
            {
                //evaluate the root to leaf decomposition provided we aren't at the leaf node and update the mean field hamiltonian
                CALL_AND_HANDLE(r2l_core::evaluate(m_ortho_engine, m_nodes[m_orthogonality_centre], m_U, bond_index), "Failed to compute the root to leaf decomposition for a node.");
            }

            real_type be = 0.0;
            for(size_type i = 0; i < m_orthog.eng().Shost().size(); ++i)
            {
                real_type si = m_orthog.eng().Shost()(i, i);
                be += si*si*std::log(si*si);
            }
            return be;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to shift orthogonality centre.");
        }
        */
    }

    real_type compute_maximum_bond_entropy()
    {
/*  
        try
        {
            //first we ensure that the ttn is orthogonalised to the root node
            CALL_AND_HANDLE(this->orthogonalise(), "Failed to orthogonalise ttn object.");

            //now we perform an euler tour of the tree structure and truncate each bond on our first pass through ultimately shifting the orthogonality centre to the root of the tree.
            if(!m_euler_tour_initialised)
            {
                traversal_path::initialise_euler_tour(*this, m_euler_tour);
            }

            m_maximum_bond_entropy = 0;

            m_euler_tour.reset_visits();
            //now perform the euler tour
            for(size_type id : m_euler_tour)
            {
                size_type times_visited = m_euler_tour.times_visited(id);
                m_euler_tour.visit(id);

                const auto& A = m_nodes[id];
                //now provided this isn't the first time we've traversed the node we will need to apply a root to leaf node decomposition to 
                //it so that we can propagate factors down the tree structure to its children.
                if(!m_euler_tour.last_visit(id))
                {
                    //get the index of the child we will be performing the decomposition for
                    size_type mode = times_visited;

                    //if we aren't at the leaf node we shift the orthogonality centre down the correct node
                    if(!A.is_leaf())
                    {
                        ASSERT(id == m_orthogonality_centre, "Something went wrong when performing euler tour.");
                        this->shift_orthogonality_centre(mode + (id == 0 ? 0 : 1), 0.0, 0, false, true);
                    }
                }
                //if it is our final time accessing the node we shift the orthogonality centre back up the tree
                else
                {
                    if(!A.is_root())
                    {
                        ASSERT(id == m_orthogonality_centre, "Something went wrong when performing euler tour final.");
                        this->shift_orthogonality_centre(0, 0.0, 0, false, true);
                    }
                }
            }
            m_euler_tour.reset_visits();
            return m_maximum_bond_entropy;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to truncate ttn object.");
        }
*/
    }
 
  #ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn object.  Error when serialising the base object.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn object.  Error when serialising the base object.");
    }
#endif

public:
    template <typename bType> 
    static void flatten(const std::vector<bType>& buf, linalg::vector<T, backend>& res)
    {
        size_t size = 0; 
        for(size_t i = 0; i < buf.size(); ++i)
        {
            size += buf[i].size();
        }
        res.resize(size);
        size_t iskip = 0;
        for(size_t i = 0; i < buf.size(); ++i)
        {
            backend::copy(buf[i].buffer(), buf[i].size(), res.buffer() + iskip);
            iskip += buf[i].size();
        }
    }

    template <typename bType> 
    static void flatten(const std::vector<bType>& buf, linalg::reinterpreted_tensor<T, 1, backend>& res)
    {
        size_t size = 0; 
        for(size_t i = 0; i < buf.size(); ++i)
        {
            size += buf[i].size();
        }
        ASSERT(res.size() == size, "Cannot flatten array invalid size.");
        size_t iskip = 0;
        for(size_t i = 0; i < buf.size(); ++i)
        {
            backend::copy(buf[i].buffer(), buf[i].size(), res.buffer() + iskip);
            iskip += buf[i].size();
        }
    }

    template <typename ftype, typename bType> 
    static void unpack(const ftype& res, std::vector<bType>& buf)
    {
        size_t size = 0; 
        for(size_t i = 0; i < buf.size(); ++i)
        {
            size += buf[i].size();
        }
        ASSERT(res.size() == size, "Failed to unpack flattened array into multiset type.  Incompatibles sizes.");

        size_t iskip = 0;
        for(size_t i = 0; i < buf.size(); ++i)
        {
            backend::copy(res.buffer()+iskip, buf[i].size(), buf[i].buffer());
            iskip += buf[i].size();
        }
    }
};

template <typename T, typename backend> 
std::ostream& operator<<(std::ostream& os, const ms_ttn<T, backend>& t)
{
    os << "dims: [";
    for(size_t i = 0; i<t.nmodes(); ++i){os << t.dim(i) << (i+1 != t.nmodes() ? ", " : "]");}
    os << std::endl << static_cast<const tree<multiset_node_data<T, backend> >&>(t);
    return os;
}

template <typename T, typename backend>
using multiset_httensor = ms_ttn<T, backend>;

}   //namespace ttns

#include "multiset_ttn_slice.hpp"

#endif  // TTNS_MS_TTN_HPP //


