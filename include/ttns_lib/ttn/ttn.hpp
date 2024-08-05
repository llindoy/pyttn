#ifndef TTNS_TTN_HPP
#define TTNS_TTN_HPP

#include <random>
#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

#include "ttn_nodes/ms_ttn_node.hpp"
#include "ttn_nodes/ttn_node.hpp"
#include "../op.hpp"
#include "../operators/site_operators/site_operator.hpp"

#include "ttnbase.hpp"
#include "ms_ttn.hpp"

namespace ttns
{

template <typename T, typename backend>
using ttn_node = typename tree<ttn_node_data<T, backend> >::node_type;

template <typename T, typename backend = blas_backend>
class ttn : public ttn_base<ttn_node_data, T, backend> 
{
public:
    using base_type = ttn_base<ttn_node_data, T, backend>;
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
    ttn() : base_type(){}

    ttn(const ttn& other) try : base_type(other) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct TTN object.");
    }

    template <typename U, typename = typename std::enable_if<not std::is_same<T, U>::value, void>::type> 
    ttn(const ttn<U, backend>& other) try : base_type(other){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct TTN object.");
    }


    template <typename U, typename be, bool CONST>
    ttn(multiset_ttn_slice<U, be, CONST> other) : base_type()
    {
        CALL_AND_RETHROW(tree_type::construct_topology(static_cast<const typename ms_ttn<U, be>::tree_type&>(other.obj()))); 
        for(auto z : common::zip(m_nodes, other.obj()))
        {
            CALL_AND_HANDLE(std::get<0>(z)() = std::get<1>(z)()[other.slice_index()], "Failed when assigning slice index.");
        }
        
        this->m_dim_sizes = other.obj().mode_dimensions(); 
        this->m_leaf_indices = other.obj().leaf_indices();
        this->m_nset = 1;

        this->m_orthog.clear();

        this->m_orthogonality_centre = other.obj().orthogonality_centre();
        this->m_has_orthogonality_centre = other.obj().has_orthogonality_centre();

        this->m_euler_tour = other.obj().euler_tour();
        this->m_euler_tour_initialised = other.obj().euler_tour_initialised();
    }


    //template <typename U>
    //ttn(const tree<sttn_node_data>& sTTN) : base_type();
    //{   
    //    CALL_AND_HANDLE(this->operator=(sTTN), "Failed to construct ttn object from sparse ttn node information.");
    //}


    template <typename INTEGER, typename Alloc>
    ttn(const ntree<INTEGER, Alloc>& topology, bool purification = false) try : base_type(topology, 1, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct TTN object.");
    }

    template <typename INTEGER, typename Alloc>
    ttn(const ntree<INTEGER, Alloc>& topology, const ntree<INTEGER, Alloc>& capacity, bool purification = false)try : base_type(topology, capacity, 1, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct TTN object.");
    }

    ttn(const std::string& _topology, bool purification = false) try : base_type(_topology, 1, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct TTN object.");
    }

    ttn(const std::string& _topology, const std::string& _capacity, bool purification = false) try : base_type(_topology, _capacity, 1, purification) {}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct TTN object.");
    }

    size_type nset() const noexcept{return 1;}
public:
    ttn& operator=(const ttn& other) = default;
    template <typename U, typename be>
    ttn& operator=(const ttn<U, be>& other){CALL_AND_RETHROW( base_type::operator=(other));return *this;}

    //assign ttn from a multiset ttn slice
    template <typename U, typename be, bool CONST>
    ttn& operator=(multiset_ttn_slice<U, be, CONST> other)
    {
        //if these are all the same size then we just do the simple assignment operator
        if(has_same_structure(other.obj(), *this) && other.obj().mode_dimensions() == this->mode_dimensions())
        {
            bool all_fit = true;
            //first check to see if the current structure can fit the assigned structure.  If it can then we don't have any problems
            for(auto z : common::zip(m_nodes, other.obj()))
            {
                if(!std::get<0>(z).can_fit_node(std::get<1>(z)()[other.slice_index()])){all_fit = false;}
            }

            for(auto z : common::zip(m_nodes, other.obj()))
            {
                CALL_AND_HANDLE(std::get<0>(z)() = std::get<1>(z)()[other.slice_index()], "Failed when assigning slice index.");
            }
            if(!all_fit){this->m_orthog.clear();}

            this->m_orthogonality_centre = other.obj().orthogonality_centre();
            this->m_has_orthogonality_centre = other.obj().has_orthogonality_centre();
        }
        else
        {
            this->clear();
            CALL_AND_RETHROW(tree_type::construct_topology(static_cast<const typename ms_ttn<U, be>::tree_type&>(other.obj()))); 
            for(auto z : common::zip(m_nodes, other.obj()))
            {
                CALL_AND_HANDLE(std::get<0>(z)() = std::get<1>(z)()[other.slice_index()], "Failed when assigning slice index.");
            }

            this->m_dim_sizes = other.obj().mode_dimensions(); 
            this->m_leaf_indices = other.obj().leaf_indices();
            this->m_nset = 1;

            this->m_orthog.clear();

            this->m_orthogonality_centre = other.obj().orthogonality_centre();
            this->m_has_orthogonality_centre = other.obj().has_orthogonality_centre();

            this->m_euler_tour = other.obj().euler_tour();
            this->m_euler_tour_initialised = other.obj().euler_tour_initialised();
        }
        return *this;
    }

    //TODO: set up initialisation form sparse ttn object
    //template <typename U>
    //ttn& operator=(const tree<std::vector<sttn_node_data>& sTTN)
    //{   

    //}


    size_type maximum_bond_dimension() const
    {
        size_type mbd = 0;
        for(const auto& n : m_nodes)
        {
            if(n.hrank() > mbd){mbd = n.hrank();}
        }
        return mbd;
    }

    size_type minimum_bond_dimension() const
    {
        size_type mbd = 0;
        bool first_call = true;
        for(const auto& n : m_nodes)
        {
            if(!first_call)
            {
                if(n.hrank() < mbd || mbd == 0){mbd = n.hrank();}
            }
            else{first_call = false;}
        }
        return mbd;
    }

public:
    template <typename int_type> 
    void set_state(const std::vector<int_type>& si){CALL_AND_RETHROW(this->_set_state(si, 0, false));}

    template <typename int_type> 
    void set_state_purification(const std::vector<int_type>& si){CALL_AND_RETHROW(this->_set_state(si, 0, true));}

    template <typename U, typename be> 
    void set_product(const std::vector<linalg::vector<U, be> >& ps){CALL_AND_RETHROW(this->_set_product(ps));}

    template <typename Rvec> 
    void sample_product_state(std::vector<size_t>& state, const std::vector<Rvec>& relval){CALL_AND_RETHROW(this->_sample_product_state(state, relval));}

    void set_identity_purification()
    {
        CALL_AND_RETHROW(this->_set_purification());
    }
public:
    
    real_type bond_entropy(size_t /* bond_index */)
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
        return 0;
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
        return 0;
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

protected:
    void _setup_orthogonality_1bop(size_type index, bool shift_orthogonality = true)
    {
        if(shift_orthogonality)
        {
            CALL_AND_HANDLE(this->set_orthogonality_centre(index), "Failed to apply one body operator.  Failed to shift orthogonality centre.");
        }
        else{m_has_orthogonality_centre = false;}
    }

    void _apply_one_body_operator(const linalg::matrix<T, backend>& M, size_type index, bool shift_orthogonality = true)
    {
        CALL_AND_RETHROW(_setup_orthogonality_1bop(index, shift_orthogonality));

        try
        {
            //now we apply the operator to the state
            linalg::matrix<T, backend> temp = M*m_nodes[index]().as_matrix();
            m_nodes[index]().as_matrix() = temp;
        }
        catch(const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply one body operator.  Error when contracting one body operator onto state.");
        }
    }

    template <typename OpType>
    typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type 
    _apply_one_body_operator(OpType& op, size_type index, bool shift_orthogonality = true)
    {
        CALL_AND_RETHROW(_setup_orthogonality_1bop(index, shift_orthogonality));

        try
        {
            //now we apply the operator to the state
            linalg::matrix<T, backend> temp = m_nodes[index]().as_matrix();
            op.apply(temp, m_nodes[index]().as_matrix());
        }
        catch(const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply one body operator.  Error when contracting one body operator onto state.");
        }
    }
    void _apply_one_body_operator(std::shared_ptr<ops::primitive<T, backend>> op, size_type index, bool shift_orthogonality = true)
    {
        CALL_AND_RETHROW(_setup_orthogonality_1bop(index, shift_orthogonality));

        try
        {
            //now we apply the operator to the state
            linalg::matrix<T, backend> temp = m_nodes[index]().as_matrix();
            op->apply(temp, m_nodes[index]().as_matrix());
        }
        catch(const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply one body operator.  Error when contracting one body operator onto state.");
        }
    }


protected:
    void shift_dangling_bond_up(linalg::tensor<T, 3, backend>& Top, linalg::tensor<T, 4, backend>& temp, linalg::tensor<T, 4, backend>& temp2, size_type curr, size_type prev, real_type tol = real_type(0), size_type nchi=0)
    {
        //get the parent tensor as a rank 3 tensor with the bond connecting the curr and prev sites separated off
        auto ap = m_nodes[curr]().as_rank_3(m_nodes[prev].child_id());

        //contract T up into the ap tensor along the bond
        temp = linalg::tensordot(ap, Top, {1}, {0});

        //now we swap the last two indices of the temp array into temp2
        temp2 = linalg::transpose(temp, {0, 1, 3, 2});
    
        auto d = temp2.shape();

        //now get a matrix representation of this tensor
        auto mtemp = temp2.reinterpret_shape(d[0]*d[1]*d[2], d[3]);

        //U*S becomes the new parent tensor which we store in Top and V^\dagger becomes the child tensor which we put
        //into the tensor network
    }

    void shift_dangling_bond_down(linalg::tensor<T, 3, backend>& Top, linalg::tensor<T, 4, backend>& temp, linalg::tensor<T, 4, backend>& temp2, size_type curr, size_type prev, real_type tol = real_type(0), size_type nchi=0)
    {
        //get the parent tensor as a rank 3 tensor with the bond connecting the curr and prev sites separaetd off
        auto ap = m_nodes[prev]().as_rank_3(m_nodes[curr].child_id());
        auto d = ap.shape();

        //now use this object to appropriately resize the Top matrix noting that the first 2 indices of T 
        //have the same shape as ap and the final index of T is the one we aim to transfer
        auto Tt = Top.reinterpret_shape(d[0], d[1], d[2], Top.shape(2));

        //now contract the second index of Tt into the child node of this and reshape this so that we 
        temp = linalg::tensordot(Tt, m_nodes[curr], {1}, {0});

        //now we swap the last two indices of the temp array into temp2
        temp2 = linalg::transpose(temp, {0, 1, 3, 2});

        auto d2 = temp2.shape();

        //now get the correct matrix representation of this tensor
        auto mtemp = temp2.reinterpret_shape(d2[0]*d2[1], d2[2]*d2[3]);

        //U becomes the new parent tensor which we store in the parent tensor and SV^\dagger becomes 
        //the child tensor which we put into T
    }

    //a function used for evaluating the action of a MPO like tensor network on the ttn.  
    //This function handles the result of shifting the dangling bond that arises after acting
    //the left most tensor of the MPO on the ttn through the ttn until we reach the location where
    //it next needs to act
    //this assumes that T is the tensor at site i1 with the dangling bond contracted into it
    void shift_dangling_bond(linalg::tensor<T, 3, backend>& Top, size_type i1, size_type i2, real_type tol = real_type(0), size_type nchi=0)
    {
        //get the path through the tensor network connecting nodes i1 and i2
        std::list<size_type> path;  this->path(i1, i2, path);

        //now we iterate through the path - contracting the T tensor into the next tensor along the path
        //before performing a reshape and svd to shift the dangling bond along this pathway.
        size_type prev = 0;
        linalg::tensor<T, 4, backend> temp;
        linalg::tensor<T, 4, backend> temp2;

        for(size_type i : path)
        {
            //we don't contract in the first step, but we do update the index of the previous operator.
            if(i != i1)
            {
                //if the previous node is a child of the next node - that is we are still moving up
                //the tree structure then we perform the upwards contraction step
                if(m_nodes[prev].is_child_of(i))
                {
                    CALL_AND_HANDLE(
                        shift_dangling_bond_up(Top, temp, temp2, i, prev, tol, nchi),  
                        "Failed to shift dangling bond up."
                    );
                }
                //otherwise if the current node is a child of the previous node - we are heading down the tree
                else if(m_nodes[i].is_child_of(prev))
                {
                    CALL_AND_HANDLE(
                        shift_dangling_bond_down(Top, temp, temp2,i, prev, tol, nchi), 
                        "Failed to shift dangling bond down."
                    );
                }
                else
                {
                    RAISE_EXCEPTION("Failed to shift dangling bond two sites in the path were not connected.");
                }
                //now check if mode i is a child of mode i1
            }
            prev = i;
        }
    }

    //a function for applying a two-body operator to a tensor network.  The implementation provided here
    //performs this contraction using a two site approach  
    void _apply_two_body_operator(const linalg::tensor<T, 3, backend>& A, const linalg::tensor<T, 3, backend>& B, size_type i1, size_type i2, real_type tol = real_type(0), size_type nchi=0)
    {
        CALL_AND_HANDLE(this->set_orthogonality_centre(i1), "Failed to apply one body operator.  Failed to shift orthogonality centre.");
    }

public:
    ttn& apply_one_body_operator(const linalg::matrix<T, backend>& op, size_type index, bool shift_orthogonality = true)
    {
        ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
        ASSERT(op.shape(0) == m_dim_sizes[index] && op.shape(1) == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");
        
        CALL_AND_RETHROW(_apply_one_body_operator(op, m_leaf_indices[index], shift_orthogonality));
        return *this;
    }

    ttn& apply_one_body_operator(const Op<T, backend>& op, bool shift_orthogonality = true)
    {
        ASSERT(op.ndim() == 1, "Failed to apply one body operator.  Operator is not one body.");
        ASSERT(op.indices()[0] < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
        ASSERT(op.dims()[0] == m_dim_sizes[op.indices()[0]], "Failed to apply one body operator to ttn. Incompatible dimensions.");
        
        CALL_AND_RETHROW(_apply_one_body_operator(op(), m_leaf_indices[op.indices()[0]], shift_orthogonality));
        return *this;
    }    
    
    template <typename OpType>
    typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, ttn&>::type 
    apply_one_body_operator(OpType& op, size_type index, bool shift_orthogonality = true)
    {
        ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
        ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");
        
        CALL_AND_RETHROW(_apply_one_body_operator(op, m_leaf_indices[index], shift_orthogonality));
        return *this;
    }

    ttn& apply_one_body_operator(site_operator<T, backend>& op, bool shift_orthogonality = true)
    {
        size_type index = op.mode();
        ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
        ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");
        
        CALL_AND_RETHROW(_apply_one_body_operator(op.op(), m_leaf_indices[index], shift_orthogonality));
        return *this;
    }

    ttn& apply_one_body_operator(site_operator<T, backend>& op, size_type index, bool shift_orthogonality = true)
    {
        ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
        ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");
        
        CALL_AND_RETHROW(_apply_one_body_operator(op.op(), m_leaf_indices[index], shift_orthogonality));
        return *this;
    }

    ttn& apply_one_body_operator(std::shared_ptr<ops::primitive<T, backend>> op, size_type index, bool shift_orthogonality = true)
    {
        ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
        ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");
        
        CALL_AND_RETHROW(_apply_one_body_operator(op, m_leaf_indices[index], shift_orthogonality));
        return *this;
    }

    //allow for a generic operator type object.
    //ttn& apply_one_body_operator
    ttn& apply_operator(const Op<T, backend>& op, real_type tol = real_type(0), size_type nchi=0)
    {
        //first check that the operator is consistent with the TTN we are acting on
        for(size_type i = 0; i < op.ndim(); ++i)
        {
            ASSERT(op.indices()[i] < this->nmodes(), "Failed to apply operator to ttn. Index out of bounds.");
        }
        for(size_type i = 0; i < op.ndim(); ++i)
        {
            ASSERT(op.dims()[i] == m_dim_sizes[op.indices()[i]], "Failed to apply operator to ttn. Incompatible dimensions.");
        }

        //currently we only support the application of one and two body operator.  3 body and above operators
        //will require decomposition of a general operator into a TTNO with the structure of the subtree
        //containing the nodes it acts on and the entire path up to their lowest common ancestor
        if(op.ndim() == 1)
        {
            CALL_AND_RETHROW(_apply_one_body_operator(op(), m_leaf_indices[op.indices()[0]], true));
        }
        else if(op.ndim() == 2)
        {
        }
        else
        {
            RAISE_EXCEPTION("N>2 body operator applications are currently not supported.");
        }
        return *this;
    }

public:
    //here the collapse algorithm will be implemented in place.  This will be done iteratively shifting the orthogonality centre of the tree to a leaf.  Computing the probability of observing the state in each possible configuration of that leaf.  Then sampling the state based on this.  
    real_type collapse(std::vector<size_t>& state, bool truncate=false, real_type tol = real_type(0), size_type nchi = 0)
    {
        state.resize(m_nleaves);
        real_type pitot = 1.0;
        this->orthogonalise();
        this->normalise();
        for(size_t i = 0; i < m_nleaves; ++i)
        {
            std::vector<real_type> pi(m_dim_sizes[i]);
            //shift orthogonality centre to leaf index
            measure_without_collapse(i, pi);

            real_type pisum = 0.0;
            for(size_t j = 0; j < m_dim_sizes[i]; ++j)
            {
                pisum += pi[j];
            }

            //now sample from the projection expectation values
            std::discrete_distribution<std::size_t> d{pi.begin(), pi.end()};
            auto& A = m_nodes[m_leaf_indices[i]];
            auto& a = A().as_matrix();
            size_t ind = d(_rng);
            state[i] = ind;

            //now that we have sampled the index to retain we need to collapse the state onto this index.
            for(size_t j=0; j < m_dim_sizes[i]; ++j)
            {
                //if we aren't in the measured state send to zero
                if(j != ind){a[j] *= 0.0;}
                //otherwise divide by the probability of observing this state to ensure correct normalisation.
                else{a[j] /= std::sqrt(pi[j]);}
            }
            pitot *= (pi[ind]/pisum);
            this->force_set_orthogonality_centre(m_leaf_indices[i]);
        }
        //after having collapsed each state.  We can now go through and ensure normalisation and truncate.  Due to the projective measurement this should be bond-dimension 1
        this->set_orthogonality_centre(0);
        this->normalise();
        if(truncate)
        {
        //    this->truncate(tol, nchi, rel_truncate);
            this->normalise();
        }

        return pitot;
    }


    real_type collapse_basis(std::vector<linalg::matrix<T>>& U, std::vector<size_t>& state, bool truncate=false, real_type tol = real_type(0), size_type nchi = 0)
    {
        ASSERT(U.size() == m_nleaves, "Failed to collapse in user specified basis.  Basis transformation vectors are not compatible with ");
        state.resize(m_nleaves);
        real_type pitot = 1.0;
        this->orthogonalise();
        this->normalise();
        for(size_t i = 0; i < m_nleaves; ++i)
        {
            //for each basis we construct a random
            std::vector<real_type> pi(m_dim_sizes[i]);
            //shift orthogonality centre to leaf index
            this->set_orthogonality_centre(m_leaf_indices[i]);

            linalg::matrix<T> b = U[i]*m_nodes[m_leaf_indices[i]]().as_matrix();

            real_type pisum = 0.0;
            for(size_t j = 0; j < m_dim_sizes[i]; ++j)
            {
                pi[j] = linalg::real(linalg::dot_product(linalg::conj(b[j]), b[j]));
                pisum += pi[j];
            }

            //now sample from the projection expectation values
            std::discrete_distribution<std::size_t> d{pi.begin(), pi.end()};
            size_t ind = d(_rng);
            state[i] = ind;

            //now that we have sampled the index to retain we need to collapse the state onto this index.
            for(size_t j=0; j < m_dim_sizes[i]; ++j)
            {
                //if we aren't in the measured state send to zero
                if(j != ind){b[j] *= 0.0;}
                //otherwise divide by the probability of observing this state to ensure correct normalisation.
                else{b[ind] /= std::sqrt(pi[ind]);}
            }
            m_nodes[m_leaf_indices[i]]().as_matrix() = linalg::adjoint(U[i])*b;
            pitot *= (pi[ind]/pisum);
            this->force_set_orthogonality_centre(m_leaf_indices[i]);
        }
        //after having collapsed each state.  We can now go through and ensure normalisation and truncate.  Due to the projective measurement this should be bond-dimension 1
        this->set_orthogonality_centre(0);
        this->normalise();

        if(truncate)
        {
        //    this->truncate(tol, nchi, rel_truncate);
            this->normalise();
        }

        return pitot;
    }

    //function for measuring a single qubit
    void measure_without_collapse(size_type i, std::vector<real_type>& res)
    {
        ASSERT(i < m_nleaves, "Cannot measure on requested mode.  Index out of bounds.");
        res.resize(m_dim_sizes[i]);
        //shift orthogonality centre to leaf index
        this->set_orthogonality_centre(m_leaf_indices[i]);

        const auto& A = m_nodes[m_leaf_indices[i]];
        const auto& a = A().as_matrix();
        for(size_t j = 0; j < m_dim_sizes[i]; ++j)
        {
            res[j] = linalg::real(linalg::dot_product(linalg::conj(a[j]), a[j]));
        }
    }

    //function for performing a measurement on all modes.
    void measure_all_without_collapse(std::vector<std::vector<real_type>>& res)
    {
        res.resize(m_nleaves);
        for(size_t i = 0; i < m_nleaves; ++i)
        {
            measure_without_collapse(i, res[i]);
        }
    }


};

template <typename T, typename backend, typename real_type = typename linalg::get_real_type<T>::type>
real_type collapse_wavefunction(const ttn<T, backend>& o, ttn<T, backend>& res, std::vector<size_t>& state, bool truncate=false, real_type tol = real_type(0), typename backend::size_type nchi = 0, bool rel_truncate = false, bool compute_be = false)
{
    //first we copy the res array into o
    res = o;

    //now perform the inplace collapse on o
    return o.collapse(state, truncate, tol, nchi, rel_truncate, compute_be);
}

template <typename T, typename backend> 
std::ostream& operator<<(std::ostream& os, const ttn<T, backend>& t)
{
    os << "dims: [";
    for(size_t i = 0; i<t.nmodes(); ++i){os << t.dim(i) << (i+1 != t.nmodes() ? ", " : "]");}
    os << std::endl << static_cast<const tree<ttn_node_data<T, backend> >&>(t);
    return os;
}

template <typename T, typename backend>
using httensor = ttn<T, backend>;
}   //namespace ttns

#endif  // TTNS_TTN_HPP //


