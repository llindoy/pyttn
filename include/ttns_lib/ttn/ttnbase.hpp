#ifndef TTNS_TTN_BASE_HPP
#define TTNS_TTN_BASE_HPP

#include <random>
#include <map>
#include <vector>

#include <linalg/utils/genrandom.hpp>

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

#include "ttn_nodes/ms_ttn_node.hpp"
#include "ttn_nodes/ttn_node.hpp"

#include "tree/ntree_builder.hpp"

#include "sweeping/sweeping_path.hpp"
#include "orthogonality/decomposition_engine.hpp"
#include "orthogonality/root_to_leaf_decomposition.hpp"
#include "orthogonality/leaf_to_root_decomposition.hpp"

namespace ttns
{

template <typename T, typename backend, bool CONST> class multiset_ttn_slice;

template <typename size_type>
struct level_pair_comp
{
    bool operator()(const std::pair<size_type, size_type>& lhs, const std::pair<size_type, size_type>& rhs) const
    {
        if(lhs.second == rhs.second)
        {
            return lhs.first > rhs.first;
        }
        return lhs.second > rhs.second;
    }
};

template <template <typename, typename> class node_class, typename T, typename backend = linalg::blas_backend>
class ttn_base : public tree<node_class<T, backend> > 
{
public:
    static_assert(linalg::is_number<T>::value, "The first template argument to the ttn object must be a valid number type.");
    static_assert(linalg::is_valid_backend<backend>::value, "The second template argument to the ttn object must be a valid backend.");

    using real_type = typename linalg::get_real_type<T>::type;

    using matrix_type = linalg::matrix<T, backend>;
    using base_type = tree<node_class<T, backend> >;

    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using size_type = typename backend::size_type;

    using node_reference = typename base_type::node_reference;
    using const_node_reference = typename base_type::const_node_reference;

    using tree_type = typename base_type::tree_type;
    using tree_reference = typename base_type::tree_reference;
    using const_tree_reference = typename base_type::const_tree_reference;

    using node_type = typename base_type::node_type;
    using bond_matrix_type = typename node_type::bond_matrix_type;
    using population_matrix_type = typename node_type::population_matrix_type;
    using orthogonality_type = typename node_type::orthogonality_type;
    using value_type = typename node_type::value_type;

    using hrank_info = std::map<std::pair<size_t, size_t>, typename node_type::hrank_type>;

    using ancestor_index = std::set<std::pair<size_type, size_type>, level_pair_comp<size_type>>;

    template <template <typename, typename> class nc, typename U, typename be> friend class ttn_base;
    friend ttn<T, backend>;
    friend ms_ttn<T, backend>;
private:
    //provide access to base class operators
    using base_type::m_nodes;
    using base_type::m_nleaves;

    std::vector<size_type> m_dim_sizes;
    std::vector<size_type> m_dim_sizes_lhd;
    std::vector<size_type> m_leaf_indices;

    orthogonality_type m_orthog;

    size_type m_orthogonality_centre = 0;
    bool m_has_orthogonality_centre = false; 

    size_type m_nset=1;
    size_type m_nset_lhd=1;

    real_type m_maximum_bond_entropy;

    sweeping::traversal_path m_euler_tour;
    bool m_euler_tour_initialised = false;

    linalg::random_engine<linalg::blas_backend> m_hrengine;
    linalg::random_engine<backend> m_rengine;

    bool m_purification = false;
public: 
    const orthogonality_type& orthogonality_engine() const{return m_orthog;}
    orthogonality_type& orthogonality_engine(){return m_orthog;}

public:
    using base_type::size;

public:
    ttn_base() : base_type(), m_dim_sizes(), m_orthog(){}

    ttn_base(const ttn_base& other) : ttn_base()
    {
        CALL_AND_RETHROW(_assign_ttn(other)); 
    }

    template <typename U, typename = typename std::enable_if<not std::is_same<T, U>::value, void>::type> 
    ttn_base(const ttn_base<node_class, U, backend>& other) : ttn_base()
    {
        CALL_AND_RETHROW(_assign_ttn(other)); 
    }

    template <typename INTEGER, typename Alloc>
    ttn_base(const ntree<INTEGER, Alloc>& topology, size_type nset = 1, bool purification = false) : m_nset(nset), m_purification(purification)
    {
        CALL_AND_HANDLE(construct_topology(topology, nset), "Failed to construct the ttn object.  Failed to allocate tree structure from topology ntree.");
    }

    template <typename INTEGER, typename Alloc>
    ttn_base(const ntree<INTEGER, Alloc>& topology, const ntree<INTEGER, Alloc>& capacity, size_type nset = 1, bool purification = false) : m_nset(nset), m_purification(purification)
    {
        CALL_AND_HANDLE(construct_topology(topology, capacity, nset), "Failed to construct the ttn object.  Failed to allocate tree structure from topology ntree.");
    }

    ttn_base(const std::string& _topology, size_type nset = 1, bool purification = false) : m_nset(nset), m_purification(purification)
    {
        ntree<size_type> topology(_topology);
        CALL_AND_HANDLE(construct_topology(topology, nset), "Failed to construct the ttn object.  Failed to allocate tree structure from topology ntree.");
    }

    ttn_base(const std::string& _topology, const std::string& _capacity, size_type nset=1, bool purification = false) : m_nset(nset), m_purification(purification)
    {
        ntree<size_type> topology(_topology);
        ntree<size_type> capacity(_capacity);
        CALL_AND_HANDLE(construct_topology(topology, capacity, nset), "Failed to construct the ttn object.  Failed to allocate tree structure from topology ntree.");
    }

protected:
    template <typename U, typename BU>
    static void assign_node(ttn_node_data<T, backend>& r, const ttn_node_data<U, BU>& i)
    {
        r = i;
    }

    template <typename U, typename BU>
    static void assign_node(std::vector<ttn_node_data<T, backend>>& r, const std::vector<ttn_node_data<U, BU>>& i)
    {
        ASSERT(r.size() >= i.size(), "input does not fit into output.");
        for(size_t ind = 0; ind < i.size(); ++ind)
        {
            r[ind] = i[ind];
        }
    }

    template <typename U, typename be>
    void _assign_ttn(const ttn_base<node_class, U, be>& other)
    {
        //first we check whether the current ttn object is capable of fitting the other one.  If it is then we don't need to attempt a reallocation at all
        if(has_same_structure(*this, other) && other.mode_dimensions() == this->mode_dimensions() && this->nset() == other.nset() && this->is_purification() == other.is_purification())
        {
            bool all_fit = true;
            //first check to see if the current structure can fit the assigned structure.  If it can then we don't have any problems
            for(auto z : common::zip(m_nodes, other))
            {
                if(!std::get<0>(z).can_fit_node(std::get<1>(z)())){all_fit = false;}
            }

            for(auto z : common::zip(m_nodes, other))
            {
                CALL_AND_HANDLE(assign_node(std::get<0>(z)(), std::get<1>(z)()), "Failed when assigning slice index."); 
            }
            if(!all_fit){reset_orthogonality();}

            m_hrengine = other.m_hrengine;
            m_rengine = other.m_rengine;

            m_purification = other.is_purification();
            m_orthogonality_centre = other.m_orthogonality_centre;
            m_has_orthogonality_centre = other.m_has_orthogonality_centre;

            if(other.m_euler_tour_initialised)
            {
                m_euler_tour = other.m_euler_tour;
                m_euler_tour_initialised = other.m_euler_tour_initialised;
            }
        }
        
        //otherwise we reassign 
        else
        {
            clear();
            CALL_AND_RETHROW(base_type::operator=(other)); 
            m_dim_sizes = other.mode_dimensions(); 
            m_dim_sizes_lhd = other.mode_dimensions_lhd(); 
            m_leaf_indices = other.leaf_indices();
            m_nset = other.m_nset;
            m_nset_lhd = other.m_nset_lhd;

            m_orthog.clear();
            m_hrengine = other.m_hrengine;
            m_rengine = other.m_rengine;


            m_purification = other.is_purification();
            m_orthogonality_centre = other.m_orthogonality_centre;
            m_has_orthogonality_centre = other.m_has_orthogonality_centre;

            m_euler_tour = other.m_euler_tour;
            m_euler_tour_initialised = other.m_euler_tour_initialised;
        }
    }

public:
    void reset_orthogonality(){m_orthog.clear();}

    void reset_orthogonality_centre()
    {
        m_orthogonality_centre = 0;
        m_has_orthogonality_centre = false;
    }

public:
    ttn_base& operator=(const ttn_base& other)
    {
        _assign_ttn(other);
        return *this;
    }

    template <typename U, typename be, typename = typename std::enable_if<not std::is_same<T, U>::value or not std::is_same<be, backend>::value, void>::type> 
    ttn_base& operator=(const ttn_base<node_class, U, be>& other)
    {
        _assign_ttn(other);
        return *this;
    }


    template <typename INTEGER, typename Alloc>
    void resize(const ntree<INTEGER, Alloc>& topology, size_type nset = 1, bool purification = false)
    {
        CALL_AND_HANDLE(clear(), "Failed to resize ttn object.  Failed to clear currently allocated data.");
        m_purification = purification;
        CALL_AND_HANDLE(construct_topology(topology, nset), "Failed to resize the ttn object.  Failed to allocate tree structure from topology ntree.");
    }


    template <typename INTEGER, typename Alloc>
    void resize(const ntree<INTEGER, Alloc>& topology, const ntree<INTEGER, Alloc>& capacity, size_type nset=1, bool purification = false)
    {
        CALL_AND_HANDLE(clear(), "Failed to resize ttn object.  Failed to clear currently allocated data.");
        m_purification = purification;
        CALL_AND_HANDLE(construct_topology(topology, capacity, nset), "Failed to resize the ttn object.  Failed to allocate tree structure from topology ntree.");
    }


    void resize(const std::string& _topology, size_type nset=1, bool purification = false)
    {
        ntree<size_type> topology(_topology);
        CALL_AND_HANDLE(resize(topology, nset, purification), "Failed to resize the ttn object.  Failed to allocate tree structure from topology ntree.");
    }

    void resize(const std::string& _topology, const std::string& _capacity, size_type nset=1, bool purification = false)
    {
        ntree<size_type> topology(_topology);
        ntree<size_type> capacity(_capacity);
        CALL_AND_HANDLE(resize(topology, capacity, nset, purification), "Failed to resize the ttn object.  Failed to allocate tree structure from topology ntree.");
    }

    std::mt19937& rng(){return m_hrengine.rng();}
    const std::mt19937& rng() const{return m_hrengine.rng();}

    linalg::random_engine<backend>& random_engine(){return m_rengine;}
    const linalg::random_engine<backend>& random_engine() const{return m_rengine;}

    linalg::random_engine<linalg::blas_backend>& random_engine_host(){return m_hrengine;}
    const linalg::random_engine<linalg::blas_backend>& random_engine_host() const{return m_hrengine;}

    template <typename sseq>
    void set_seed(sseq& seed)
    {
        m_hrengine.set_seed(seed);
        m_rengine.set_seed(seed);
    }

    void random()
    {
        try
        {
            if(!m_orthog.is_initialised()){m_orthog.init(*this);}

            for(auto& n : reverse(m_nodes))
            {
                n.set_node_random(m_rengine);

                if(!n.is_root())
                {
                    CALL_AND_HANDLE(n.decompose_up(m_orthog), "Failed to shift orthogonality up.");
                    CALL_AND_HANDLE(n.apply_to_node(m_orthog), "Failed to shift orthogonality up.");
                }
            }
            m_orthogonality_centre = 0;
            m_has_orthogonality_centre = true;
            normalise();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise random tensors.");
        }
    }

    void zero()
    {
        for(auto& ch : reverse(m_nodes)){ch.zero();} 
        m_has_orthogonality_centre = false;
    }

protected:
    template <typename int_type> 
    void _set_state(const std::vector<int_type>& si, size_type set_index = 0, bool use_purification_info = false, bool random_unoccupied_initialisation=true)
    {
        //if we aren't using the set_state_purification function or we don't have a purification then we just set as usual
        if(!use_purification_info || !m_purification)
        {
            ASSERT(set_index < this->nset(), "Cannot set ttnbase to specified state.  Set index out of bounds.");
            ASSERT(si.size() == this->nmodes(), "Cannot set ttnbase to specified state.  The state does not have the required numbers of modes.");

            for(size_type i = 0; i < this->nmodes(); ++i)
            {
                ASSERT(static_cast<size_type>(si[i]) < m_dim_sizes[i], "Cannot set state, state index out of bounds.");
            }
            
            //now zero the state
            this->zero();
            this->initialise_logical_tensors_product_state(set_index);

            for(size_type i = 0; i < this->nmodes(); ++i)
            {
                CALL_AND_HANDLE(m_nodes[m_leaf_indices[i]].set_leaf_node_state(set_index, si[i], m_rengine, random_unoccupied_initialisation), "Failed to set state.");
            }
        }
        //otherwise we need to do a strided set the correct degrees of freedom and ensure that when we trace over the ancilla we receive the state we aimed
        //to set
        else
        {
            ASSERT(set_index < this->nset(), "Cannot set ttnbase to specified state.  Set index out of bounds.");
            ASSERT(si.size() == this->nmodes(), "Cannot set ttnbase to specified state.  The state does not have the required numbers of modes.");

            for(size_type i = 0; i < this->nmodes(); ++i)
            {
                ASSERT(static_cast<size_type>(si[i]) < m_dim_sizes_lhd[i], "Cannot set state, state index out of bounds.");
            }
            
            //now zero the state
            this->zero();
            this->initialise_logical_tensors_product_state(set_index);

            for(size_type i = 0; i < this->nmodes(); ++i)
            {
                CALL_AND_HANDLE(m_nodes[m_leaf_indices[i]].set_leaf_node_state(set_index, si[i]*(m_dim_sizes_lhd[i]), m_rengine, random_unoccupied_initialisation), "Failed to set state.");
            }
        }

        //now enforce that the orthogonality centre is at the root
        this->force_set_orthogonality_centre(0);
    }

    template <typename U, typename be> 
    void _set_product(const std::vector<linalg::vector<U, be> >& ps, size_type set_index = 0)
    {
        ASSERT(set_index < this->nset(), "Cannot set ttnbase to specified state.  Set index out of bounds.");
        ASSERT(ps.size() == this->nmodes(), "Cannot set ttn to specified state.  The state does not have the required numbers of modes.");

        for(size_type i = 0; i < this->nmodes(); ++i)
        {
            ASSERT(ps[i].size() == m_dim_sizes[i], "Cannot set state as product state the product array is the wrong size.");
        }

        //now zero the state
        this->zero();
        this->initialise_logical_tensors_product_state(set_index);

        m_has_orthogonality_centre = false;
        for(size_type i = 0; i < this->nmodes(); ++i)
        {
            m_nodes[m_leaf_indices[i]].set_leaf_node_vector(set_index, ps[i], m_rengine);
        }
        this->orthogonalise();
    }

    template <typename Rvec> 
    void _sample_product_state(std::vector<size_t>& state, const std::vector<Rvec>& relval, size_type set_index = 0)
    {
        ASSERT(set_index < this->nset(), "Cannot set ttnbase to specified state.  Set index out of bounds.");
        ASSERT(relval.size() == this->nmodes(), "Cannot set ttn to specified state.  The state does not have the required numbers of modes.");
        state.resize(this->nmodes());

        for(size_type i = 0; i < this->nmodes(); ++i)
        {
            ASSERT(relval[i].size() == m_dim_sizes[i], "Cannot set state as product state the product array is the wrong size.");
        }

        //now zero the state
        this->zero();
        this->initialise_logical_tensors_product_state(set_index);

        m_has_orthogonality_centre = false;
        for(size_type i = 0; i < this->nmodes(); ++i)
        {
            std::discrete_distribution<std::size_t> d{relval[i].begin(), relval[i].end()};
            size_type ind = d(m_hrengine.rng());
            state[i] = ind;
            CALL_AND_HANDLE(m_nodes[m_leaf_indices[i]].set_leaf_node_state(set_index, ind, m_rengine), "Failed to set state.");
        }

        //now enforce that the orthogonality centre is at the root
        this->force_set_orthogonality_centre(0);
    }

    void _set_purification(size_type set_index = 0)
    {
        //now zero the state
        this->zero();
        this->initialise_logical_tensors_product_state(set_index);

        for(size_type i = 0; i < this->nmodes(); ++i)
        {
            CALL_AND_HANDLE(m_nodes[m_leaf_indices[i]].set_leaf_purification(set_index, m_rengine), "Failed to set state.");
        }

        //now enforce that the orthogonality centre is at the root
        this->force_set_orthogonality_centre(0);
    }

protected:
    void initialise_logical_tensors_product_state(size_type set_index = 0)
    {
        for(auto& c : m_nodes)
        {
            if(c.is_leaf()){}
            //if its an interior node fill it with the identity matrix
            else{CALL_AND_HANDLE(c.set_node_identity(set_index), "Failed to set interior nodes to identity.");}
        }
    }

public:
    const bond_matrix_type& active_bond_matrix() const{return m_orthog.bond_matrix();}
    bond_matrix_type& active_bond_matrix() {return m_orthog.bond_matrix();}

    const population_matrix_type& active_population_matrix() const{return m_orthog.population_matrix();}
    population_matrix_type& active_population_matrix() {return m_orthog.population_matrix();}
public:

    void clear()
    {
        try
        {
            CALL_AND_RETHROW(base_type::clear());
            m_orthog.clear();
            m_orthogonality_centre = 0;
            m_has_orthogonality_centre = false;
            m_nset = 0;
            m_purification = false;

            m_euler_tour.clear();
            m_euler_tour_initialised = false;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear ttn object.");
        }
    }

    sweeping::traversal_path& euler_tour()
    {
        ASSERT(m_euler_tour_initialised, "Failed to access euler tour it has not been initialised.");
        return m_euler_tour;
    }

    const sweeping::traversal_path& euler_tour() const
    {
        ASSERT(m_euler_tour_initialised, "Failed to access euler tour it has not been initialised.");
        return m_euler_tour;
    }
    bool euler_tour_initialised() const{return m_euler_tour_initialised;}

    const std::vector<size_type>& mode_dimensions() const{return m_dim_sizes;}
    const std::vector<size_type>& mode_dimensions_lhd() const{return m_dim_sizes_lhd;}
    size_type dim(size_type i) const{return m_dim_sizes[i];}
    size_type nmodes() const noexcept {return m_dim_sizes.size();}
    size_type ntensors() const noexcept{return m_nodes.size();}
    size_type nset() const noexcept{return m_nset;}
    bool is_purification() const noexcept{return m_purification;}
    size_type orthogonality_centre() const noexcept{return m_orthogonality_centre;}
    real_type maximum_bond_entropy() const noexcept{return m_maximum_bond_entropy;}

    size_type nelems() const 
    {
        if(m_dim_sizes.size() == 0){return 0;}
        size_type nelems = 1;
        for(size_type i=0; i<m_dim_sizes.size(); ++i){nelems *= m_dim_sizes[i];}
        return nelems*m_nset;
    }

    bool is_orthogonalised() const
    {   
        if(m_has_orthogonality_centre)
        {
            return m_orthogonality_centre == 0;
        }
        return false;
    }

    bool has_orthogonality_centre() const{return m_has_orthogonality_centre;}


    void setup_orthogonality(){if(!m_orthog.is_initialised()){m_orthog.init(*this);}}


    void bond_dimensions(hrank_info& binfo) const
    {
        for(const auto& a : m_nodes)
        {
            if(!a.is_root())
            {
                a.get_hrank(binfo[std::make_pair(a.id(), a.parent().id())]);
            }
        }
    }

    void bond_capacities(hrank_info& binfo) const
    {
        for(const auto& a : m_nodes)
        {
            if(!a.is_root())
            {
                a.get_max_hrank(binfo[std::make_pair(a.id(), a.parent().id())]);
            }
        }
    }

    /*
     *  Functions for adjusting the orthogonality condition of the TTNS
     */
public:
    void force_set_orthogonality_centre(size_t index)
    {
        ASSERT(index < m_nodes.size(), "Failed to set orthogonality centre. Index out of bounds.");
        m_orthogonality_centre = index; m_has_orthogonality_centre = true;
    }

    void force_set_orthogonality_centre(const std::list<size_type>& index)
    {
        size_type id;
        CALL_AND_HANDLE(id = this->id_at(index), "Failed to set orthogonality centre.   Failed to find the specified node by index.");
        m_orthogonality_centre = id; 
        m_has_orthogonality_centre = true;
    }

    //function for shifting the orthogonality centre along a given bond of the current orthogonality centre.  Potentially with truncation if either the tol variable or the nchi variables are set
    //and are less than the current dimension of this bond
    void shift_orthogonality_centre(size_t bond_index, real_type tol = real_type(0), size_type nchi = 0)
    {
        try
        {
            if(!m_orthog.is_initialised()){m_orthog.init(*this);}

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
                    CALL_AND_HANDLE(m_nodes[m_orthogonality_centre].shift_orthogonality_up(m_orthog, tol, nchi), "Failed to shift orthogonality up.");
                    bond_shifted = true;

                    //if we are shifting the node up then we set that the current node is orthogonalised
                    m_orthogonality_centre = m_nodes[m_orthogonality_centre].parent_pointer()->id();
                }
                else
                {
                    --bond_index;
                }
            }
            if(!bond_shifted)
            {
                CALL_AND_HANDLE(m_nodes[m_orthogonality_centre].shift_orthogonality_down(m_orthog, bond_index, tol, nchi), "Failed to shift orthogonality down.");
                m_orthogonality_centre = m_nodes[m_orthogonality_centre].child_pointer(bond_index)->id();
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to shift orthogonality centre.");
        }
    }

    void set_orthogonality_centre(size_type index, real_type tol = real_type(0), size_type nchi = 0)
    {
        try
        {
            ASSERT(index < m_nodes.size(), "Failed to set orthogonality centre. Index out of bounds.");

            if(!m_orthog.is_initialised()){m_orthog.init(*this);}

            if(!this->has_orthogonality_centre())
            {
                CALL_AND_HANDLE(this->_orthogonalise(), "Failed to orthogonalise non-orthogonal TTN object.");
            }
            CALL_AND_HANDLE(this->_set_orthogonality_centre(index, tol, nchi), "Failed to handle TTN with orthogonality centre.  Failed to shift node to index.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to set orthogonality centre for ttn.");
        }
    }

    void set_orthogonality_centre(const std::list<size_type>& index, real_type tol = real_type(0), size_type nchi = 0)
    {
        size_type id;
        CALL_AND_HANDLE(id = this->id_at(index), "Failed to set orthogonality centre.   Failed to find the specified node by index.");
        CALL_AND_RETHROW(this->set_orthogonality_centre(id, tol, nchi));
    }

    //takes a generic TTN and shifts the orthogonality centre to the root node
    void orthogonalise(bool force = false)
    {
        try
        {
            if(!m_orthog.is_initialised()){m_orthog.init(*this);}

            if(this->has_orthogonality_centre() && !force)
            {
                CALL_AND_HANDLE(this->_set_orthogonality_centre(0), "Failed to handle TTN with orthogonality centre.  Failed to shift node to root.");
            }
            else
            {
                //here we need to perform the leaf to root decomposition.
                CALL_AND_HANDLE(this->_orthogonalise(), "Failed to orthogonalise non-orthogonal TTN object.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to orthogonalise general ttn.");
        }
    }

    real_type truncate(real_type tol = real_type(0), size_type nchi = 0)
    {
        //RAISE_EXCEPTION("Truncate currently doesn't work.");
        try
        {
            if(!m_orthog.is_initialised()){m_orthog.init(*this);}
            //first we ensure that the ttn is orthogonalised to the root node
            CALL_AND_HANDLE(this->orthogonalise(), "Failed to orthogonalise ttn object.");

            //now we perform an euler tour of the tree structure and truncate each bond on our first pass through ultimately shifting the orthogonality centre to the root of the tree.
            if(!m_euler_tour_initialised)
            {
                sweeping::traversal_path::initialise_euler_tour(*this, m_euler_tour);
            }

            m_maximum_bond_entropy = 0.0;

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
                        this->shift_orthogonality_centre(mode + (id == 0 ? 0 : 1), tol, nchi);
                    }
                }
                //if it is our final time accessing the node we shift the orthogonality centre back up the tree
                else
                {
                    if(!A.is_root())
                    {
                        ASSERT(id == m_orthogonality_centre, "Something went wrong when performing euler tour final truncate.");
                        this->shift_orthogonality_centre(0, tol, nchi);
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
    }

    real_type normalise()
    {
        try
        {
            if(!this->has_orthogonality_centre())
            {
                //first ensure orthogonalisation of the ttn object
                CALL_AND_HANDLE(this->orthogonalise(), "Failed to orthogonalise ttn object.");
            }

            //now that we have normalised the ttn object we can normalise it by simply acting with the root node
            real_type norm = m_nodes[m_orthogonality_centre].norm();

            ASSERT(std::abs(norm) != 0, "Cannot normalise matrix.  Norm is zero.");

            m_nodes[m_orthogonality_centre] /= norm;
            this->force_set_orthogonality_centre(m_orthogonality_centre);
  
            return norm;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to normalise ttn object.");
        }
    }

    void conj()
    {
        try
        {
            for(auto& a : m_nodes)
            {
                CALL_AND_RETHROW(a.conj());
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to conjugate ttn object.");
        }
    }

    real_type norm() const
    {
        real_type norm=0;
        try
        {
            if(this->has_orthogonality_centre())
            {
                norm = m_nodes[m_orthogonality_centre].norm();
            }
            else
            {
                RAISE_EXCEPTION("Norm has not yet been implemented for non-orthogonalised TTNs.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to normalise ttn object.");
        }
        return norm;
    }
public:
    //scalar inplace multiplication and division
    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, ttn_base&>::type operator*=(const U& u)
    {
        if(this->has_orthogonality_centre())
        {
            m_nodes[m_orthogonality_centre] *= u;
        }
        else
        {
            m_nodes[0] *= u;
        }
        return *this;
    }

    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, ttn_base&>::type operator/=(const U& u)
    {
        if(this->has_orthogonality_centre())
        {
            m_nodes[m_orthogonality_centre] /= u;
        }
        else
        {
            m_nodes[0]().as_matrix() /= u;
        }
        return *this;
    }

public:
    size_type get_leaf_index(size_type lid)
    {
        ASSERT(lid < m_nleaves, "Invalid leaf index.");
        return m_leaf_indices[lid];
    }
    
    const std::vector<size_type>& leaf_indices() const{return m_leaf_indices;}

    const value_type& site_tensor(size_type lid) const
    {
        ASSERT(lid < m_nleaves, "Invalid leaf index.");
        return m_nodes[m_leaf_indices[lid]]();
    }

    value_type& site_tensor(size_type lid)
    {
        ASSERT(lid < m_nleaves, "Invalid leaf index.");
        size_type index = m_leaf_indices[lid];
        if(m_has_orthogonality_centre && m_orthogonality_centre != index){m_has_orthogonality_centre = false;}
        return m_nodes[index]();
    }

    node_reference operator[](size_type i) 
    {
        if(m_has_orthogonality_centre){m_has_orthogonality_centre = (i == m_orthogonality_centre);}
        return m_nodes[i];
    }
    
    const_node_reference operator[](size_type i)  const
    {
        return m_nodes[i];
    }

    //gets the path connecting two leaf nodes and additionally returns the node associated with the root of the subtree that contains these two leaves
    size_t leaf_path(size_t li, size_t lj, std::list<size_type>& path)
    {
        ASSERT(li < m_nleaves && lj < m_nleaves, "Invalid leaf index.");

        //for the trivial case of the same leaf node the root is the current node and the path is empty
        if(li == lj){return li;}

        CALL_AND_HANDLE(return this->path(m_leaf_indices[li], m_leaf_indices[lj], path), "Failed to determine path connecting two leaf nodes.");
    }

public:
    //a function for obtaining the traversal path from node i to node j.  This also returns the index of the smallest subtree containing the entire path
    size_t path(size_t ind_i, size_t ind_j, std::list<size_type>& path)
    {
        ASSERT(ind_i < m_nodes.size() && ind_j < m_nodes.size(), "Failed to find path between nodes.  Index out of bounds.");
        path.clear();
        size_t root = ind_i;
        if(ind_i != ind_j)
        {
            const std::list<size_type>& index_i = m_nodes[ind_i].index();
            const std::list<size_type>& index_j = m_nodes[ind_j].index();
            std::list<size_type> root_ind;

            auto it = index_i.begin();
            auto jt = index_j.begin();
            size_type counter = 0;
            bool continue_iter = true;

            for(; it != index_i.end() && jt != index_j.end() && continue_iter; ++it, ++jt)
            {
                if(*it == *jt){++counter; root_ind.push_back((*it));}
                else
                {
                    continue_iter = false;
                }
            }
            if(!continue_iter && jt != index_j.begin()){--jt;}

            size_type ncommon = counter;
            size_type n_to_remove = index_i.size() - ncommon;
            root = this->id_at(root_ind);

            //move back to the index of the first common node
            for(size_type i = 0; i < n_to_remove; ++i){path.push_back(0);}


            //if(ncommon != index_j.size()){--jt;}
            for(;jt != index_j.end(); ++jt)
            {
                if(jt == index_j.begin())
                {
                    path.push_back((*jt));
                }
                else
                {
                    path.push_back((*jt)+1);
                }
            }
        }
        return root;
    }

    void ancestor_indexing(const size_type& ind, ancestor_index& inds) const
    {
        ASSERT(ind < this->ntensors(), "Cannot construct ancestor indexing for node.  Index out of bounds.");
        size_type curr_ind = ind;
        while(!m_nodes[curr_ind].is_root())
        {
            //inserting the element into the set sorted by level index.
            inds.insert(std::make_pair(curr_ind, m_nodes[curr_ind].level()));
            curr_ind = m_nodes[curr_ind].parent().id();
        }
        inds.insert(std::make_pair(curr_ind, m_nodes[curr_ind].level()));
    }

    void ancestor_indexing_leaf(const size_type& li, ancestor_index& inds) const
    {
        ASSERT(li < m_nleaves, "Leaf index out of bounds.");
        ancestor_indexing(m_leaf_indices[li], inds);
    }


    void ancestor_indexing_leaf(const std::vector<size_type>& linds, ancestor_index& inds) const
    {
        //for each leaf in the list of leaves
        for(size_type li : linds)
        {
            CALL_AND_RETHROW(ancestor_indexing_leaf(li, inds));
        }
    }
    
    void ancestor_indexing(const std::vector<size_type>& linds, ancestor_index& inds) const
    {
        //for each leaf in the list of leaves
        for(size_type li : linds)
        {
            CALL_AND_RETHROW(ancestor_indexing(li, inds));
        }
    }

protected:
    void _orthogonalise()
    {
        try
        {
            using common::rzip;
            for(auto& a : reverse(m_nodes))
            {
                m_orthog.resize_data(a);

                if(!a.is_root())
                {
                    CALL_AND_HANDLE(a.shift_orthogonality_up(m_orthog), "Failed to shift orthogonality up.");
                }
            }
            this->force_set_orthogonality_centre(0);
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("Orthogonalising the TTN object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to orthogonalise the TTN object.");
        }
    }

    void _set_orthogonality_centre(size_type index, real_type tol = real_type(0), size_type nchi = 0)
    {
        try
        {
            if(m_orthogonality_centre == index){return;}

            std::list<size_type> traversal_path;
            this->path(m_orthogonality_centre, index, traversal_path);

            for(const size_type& i : traversal_path)
            {
                CALL_AND_HANDLE(this->shift_orthogonality_centre(i, tol, nchi), "Failed when shifting orthogonality centre.");
            }

            ASSERT(m_orthogonality_centre == index, "Error: shifting completed but orthogonality centre found in an incorrect location");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to transfer orthogonality centre to index.");
        }
    }

protected:
  #ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nset", m_nset)), "Failed to serialise ttn object.  Failed to serialise its set dimension.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimensions", m_dim_sizes)), "Failed to serialise ttn object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_indices", m_leaf_indices)), "Failed to serialise ttn object.  Failed to serialise its leaf_indices.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("orthogonality_centre", m_orthogonality_centre)), "Failed to seriesalise ttn object. Failed to serialise orthogonality centre.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("has_orthogonality_centre", m_has_orthogonality_centre)), "Failed to seriesalise ttn object. Failed to serialise orthogonality centre.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nset", m_nset)), "Failed to serialise ttn object.  Failed to serialise its set dimension.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimensions", m_dim_sizes)), "Failed to serialise ttn object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_indices", m_leaf_indices)), "Failed to serialise ttn object.  Failed to serialise its leaf_indices.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("orthogonality_centre", m_orthogonality_centre)), "Failed to seriesalise ttn object. Failed to serialise orthogonality centre.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("has_orthogonality_centre", m_has_orthogonality_centre)), "Failed to seriesalise ttn object. Failed to serialise orthogonality centre.");
    }
#endif

protected:    
    template <typename INTEGER, typename Alloc>
    void construct_topology(const ntree<INTEGER, Alloc>& _tree)
    {
        CALL_AND_RETHROW(construct_topology(_tree, _tree));
    }

    template <typename INTEGER, typename Alloc>
    void construct_topology(const ntree<INTEGER, Alloc>& _tree, size_type nset=1)
    {
        CALL_AND_RETHROW(construct_topology(_tree, _tree, nset));
    }

    template <typename INTEGER, typename Alloc>
    void construct_topology(const ntree<INTEGER, Alloc>& __tree, const ntree<INTEGER, Alloc>& _capacity, size_type nset = 1)
    {
        ASSERT(__tree.size() > 2, "Failed to build ttn from topology tree.  The input topology must contain at least 3 elements.  If it contains fewer than 3 elements then this is just a vector and we won't want to use the full TTN structure.");
        ASSERT(__tree.size() == _capacity.size(), "Failed to construct ttn topology with capacity.");

        m_nset = nset;
        m_nset_lhd = nset;
        if(m_purification)
        {
            m_nset = nset*nset;
            m_nset_lhd = nset;
        }


        ntree<INTEGER, Alloc> _tree(__tree);
        ntree<INTEGER, Alloc> capacity(_capacity);

        ntree_builder<INTEGER>::sanitise_tree(_tree, true);
        ntree_builder<INTEGER>::sanitise_tree(capacity, true);

        //otherwise if the topology tree contains more than 2 more elements we will attempt to interpret it as a (hierarchical) tucker tensor
        //and solve the problem in this space.
        m_nleaves = _tree.nleaves();

        m_dim_sizes.resize(m_nleaves);
        m_dim_sizes_lhd.resize(m_nleaves);

        m_leaf_indices.resize(m_nleaves);
        //resize the m_nodes array to the correct size
        size_type required_size = _tree.size() - _tree.nleaves();
        if(m_nodes.size() != required_size){m_nodes.resize(required_size);}

        //now we can begin building the tree
        size_type count = 0;
        size_type leaf_counter = 0; 
        auto this_it = m_nodes.begin();
        typename ntree<INTEGER, Alloc>::iterator capacity_iter = capacity.begin();
        for(typename ntree<INTEGER, Alloc>::iterator tree_iter = _tree.begin(); tree_iter != _tree.end(); ++tree_iter, ++capacity_iter)
        {
            ASSERT(capacity_iter != capacity.end(), "The capacity and tree iter objects are not the same size.");
            ASSERT(tree_iter->value() <= capacity_iter->value(), "Failed to construct ttn object.  The capacity is less than the size.");

            ASSERT(tree_iter->is_leaf() == capacity_iter->is_leaf(), "The capacity and topology trees do not have the same structure.");
            //we skip the leaves of the tree object as those are used to specify the topology
            //of the layer above but themselves do not correspond to a node of the mlmctdh topology
            if(!tree_iter->is_leaf())
            {
                if(tree_iter->size() == 1)
                {
                    if(tree_iter->operator[](0).is_leaf())
                    {
                        this_it->set_leaf_index(leaf_counter);
                        m_leaf_indices[leaf_counter] = count;
                    }
                    else
                    {
                        this_it->set_leaf_index(m_nleaves);
                    }
                }
                else
                {
                    this_it->set_leaf_index(m_nleaves);
                }

                //now we determine the number of children of tree_iter that are not leaves 
                //if the node has a child which is a leaf that must be its only child
                //as otherwise this size_tree does not represent a valid topology
                size_type nchildren = 0;
                size_type _nleaves = 0;

                for(auto child_it = tree_iter->begin(); child_it != tree_iter->end(); ++child_it)
                {
                    if(!child_it->is_leaf()){++nchildren;}
                    else{++_nleaves;}
                }

                if(nchildren == 0)
                {
                    ASSERT(_nleaves == 1, "If this is an exterior node we can only handle a single leaf.");
                }
                else
                {
                    ASSERT(nchildren > 1, "Cannot handle a pure bond matrix in the ttn.");
                    ASSERT(_nleaves == 0, "If this is an interior node we cannot handle any external degrees of freedom.");
                }

                size_type ncapacity_children = 0;
                size_type capacity_nleaves = 0;
                for(auto child_it = capacity_iter->begin(); child_it != capacity_iter->end(); ++child_it)
                {
                    if(!child_it->is_leaf()){++ncapacity_children;}
                    else{++capacity_nleaves;}
                }
                ASSERT(ncapacity_children == nchildren && capacity_nleaves == _nleaves, "The capacity and topology trees do not have the same structure.");

                //now we can allocate the children array associated with this node
                this_it->m_children.resize(nchildren);
                this_it->m_id = count;
                this_it->m_level = tree_iter->level();  


                CALL_AND_HANDLE(this_it->setup_data_from_topology_node(*tree_iter, *capacity_iter, nchildren+_nleaves, nset, m_purification), "Failed to set up data from topology.");

                //now we set up the children pointers
                size_type index3 = 1;

                for(size_type child_index = 0; child_index < nchildren; ++child_index)
                {
                    this_it->m_children[child_index] = &(*(this_it + index3));
                    this_it->m_children[child_index]->m_child_id = child_index;     
                    (this_it+index3)->m_parent = &(*this_it);
                    (this_it+index3)->m_index = this_it->m_index;
                    (this_it+index3)->m_index.push_back(child_index);
                    index3 += tree_iter->operator[](child_index).subtree_size() - tree_iter->operator[](child_index).nleaves();
                }
                
                ++count;
                ++this_it;
            }
            //if we are at a leaf node then we need to store the number of functions associated with this dimension
            else
            {
                m_dim_sizes_lhd[leaf_counter] = tree_iter->value();    
                m_dim_sizes[leaf_counter] = tree_iter->value();    

                //if this is a purification then we need to double the local hilbert space dimension
                if(m_purification){m_dim_sizes[leaf_counter]*=tree_iter->value();}
                ++leaf_counter;
            }
        }

        //always initialise the euler tour
        sweeping::traversal_path::initialise_euler_tour(*this, m_euler_tour);
        m_euler_tour_initialised = true;
    }
};

}   //namespace ttns

#endif  // TTNS_TTNBASE_HPP //


