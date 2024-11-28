#ifndef TTNS_TENSOR_NODE_HPP
#define TTNS_TENSOR_NODE_HPP


#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>
#include "../tree/tree.hpp"
#include "../tree/tree_node.hpp"
#include "../tree/ntree.hpp"
#include "ttn_node_helper.hpp"

#include <vector>
#include <stdexcept>


namespace ttns
{
//Potential TODO: Add in a better way to flag whether or not a node interacts with a physical degree of freedom.  This is currently done by ensuring 
//that only leaf nodes are physical nodes and consequently requires inserting local basis rotation nodes whenever we want to have a physical degree
//of freedom.
template <typename T, typename backend = linalg::blas_backend>
class ttn_node_data : public linalg::matrix<T, backend> 
{
public:
    using matrix_type = linalg::matrix<T, backend>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

protected:
    std::vector<size_type> m_mode_dims;
    std::vector<size_type> m_mode_capacity;
    size_type m_max_hrank;
    size_type m_max_dimen;
    friend tree_node<tree_base<ttn_node_data<T, backend> > >;
    friend tree_node<tree_base<std::vector<ttn_node_data<T, backend> > > >;

public:
    ttn_node_data() : matrix_type(), m_mode_dims(), m_mode_capacity(), m_max_hrank(0), m_max_dimen(0) {}
    ttn_node_data(ttn_node_data&& o) = default;
    ttn_node_data(const ttn_node_data& o) : matrix_type()
    {
        CALL_AND_HANDLE(matrix_type::allocate(o.max_dimen(), o.max_hrank()), "Failed to copy assign ttn_node_data. Failed when applying base type copy operator.");
        CALL_AND_HANDLE(matrix_type::operator=(static_cast<const linalg::matrix<T, backend>&>(o)), "Failed to copy assign ttn_node_data. Failed when applying base type copy operator.");
        m_mode_dims = o.m_mode_dims;
        m_mode_capacity = o.m_mode_capacity;
        m_max_hrank = o.m_max_hrank;
        m_max_dimen = o.m_max_dimen;
    }

    ttn_node_data& operator=(const ttn_node_data& o) 
    {
        if(this->m_totcapacity < o.max_hrank()*o.max_dimen())
        {
            CALL_AND_HANDLE(matrix_type::reallocate(o.max_dimen(), o.max_hrank()), "Failed to copy assign ttn_node_data. Failed when applying base type copy operator.");
        }
        CALL_AND_HANDLE(matrix_type::operator=(static_cast<const linalg::matrix<T, backend>&>(o)), "Failed to copy assign ttn_node_data. Failed when applying base type copy operator.");
        m_mode_dims = o.m_mode_dims;
        m_mode_capacity = o.m_mode_capacity;
        m_max_hrank = o.m_max_hrank;
        m_max_dimen = o.m_max_dimen;
        return *this;
    }

    template <typename U, typename be>
    typename std::enable_if<not std::is_same<be, backend>::value or not std::is_same<U, T>::value, ttn_node_data&>::type 
    operator=(const ttn_node_data<U, be> & o) 
    {
        if(this->m_totcapacity< o.max_hrank()*o.max_dimen())
        {
            CALL_AND_HANDLE(matrix_type::reallocate(o.max_dimen(), o.max_hrank()), "Failed to copy assign ttn_node_data. Failed when applying base type copy operator.");
        }

        CALL_AND_HANDLE(matrix_type::operator=(static_cast<const linalg::matrix<U, be>&>(o)), "Failed to copy assign ttn_node_data. Failed when applying base type copy operator.");
        m_mode_dims = o.dims();
        m_mode_capacity = o.max_dims();
        m_max_hrank = o.max_hrank();
        m_max_dimen = o.max_dimen();
        return *this;
    }

    ttn_node_data& operator=(ttn_node_data&& o)
    {
        
        CALL_AND_HANDLE(matrix_type::operator=(std::forward<linalg::matrix<T, backend>>(o)), "Failed to move assign ttn_node_data. Failed when applying base type move operator.");
        m_mode_dims = std::move(o.m_mode_dims);
        m_mode_capacity = std::move(o.m_mode_capacity);
        m_max_hrank = o.m_max_hrank;
        m_max_dimen = o.m_max_dimen;
        return *this;
    }

    template <typename U, typename ube>
    typename std::enable_if<std::is_convertible<U, T>::value, ttn_node_data&>::type operator=(const linalg::matrix<U, ube>& mat)
    {   
        ASSERT(mat.shape() == matrix_type::shape(), "Failed to copy assign ttn_node_data from matrix.  The matrix is not the correct size.")
        CALL_AND_HANDLE(matrix_type::operator=(mat), "Failed to copy assign ttn_node_data from matrix. Failed when applying base type copy operator.");
        return *this;
    }

    template <typename U, typename be>
    bool can_fit_node(const ttn_node_data<U, be> & o) const
    {
        if(max_hrank() < o.max_hrank()){return false;}
        if(max_dims().size() != o.max_dims().size()){return false;}
        for(size_type i = 0; i < max_dims().size(); ++i)
        {
            if(max_dim(i) < o.max_dim(i)){return false;}
        }
        return true;
    }

    void resize(size_type hrank, const std::vector<size_type>& mode_dims)
    {
        m_mode_capacity.resize(mode_dims.size());
        size_type ndimen = 1;
        for(size_type i = 0; i < mode_dims.size(); ++i)
        {
            if(m_mode_capacity[i] < mode_dims[i]){m_mode_capacity[i] = mode_dims[i];}
            ndimen *= mode_dims[i];
            m_mode_dims[i] = mode_dims[i];
        }
        matrix_type::resize(ndimen, hrank);

        if(ndimen > m_max_dimen){m_max_dimen = ndimen;}
        if(hrank > m_max_hrank){m_max_hrank = hrank;}
    }

    void reallocate(size_type max_hrank, const std::vector<size_type>& max_mode_dims)
    {
        m_mode_capacity = max_mode_dims;    
        m_mode_dims.resize(max_mode_dims.size());
        size_type ndimen = 1;
        for(size_type i = 0; i < max_mode_dims.size(); ++i)
        {
            ndimen *= max_mode_dims[i];
        }
        m_max_hrank = max_hrank;
        m_max_dimen = ndimen;
        matrix_type::reallocate(ndimen, max_hrank);
    }

    size_type nmodes() const{return m_mode_dims.size();}

    size_type hrank(bool use_max_dim = false) const
    {
        if(!use_max_dim){return this->shape(1);}
        else{return m_max_hrank;}
    }
    size_type dimen(bool use_max_dim = false) const 
    {
        if(!use_max_dim){return this->shape(0);}
        else{return this->m_max_dimen;}
    }
    size_type dim(size_type n, bool use_max_dim = false) const
    {
        if(!use_max_dim){return m_mode_dims[n];}
        else{return m_mode_capacity[n];}
    }
    const std::vector<size_type>& dims() const
    {
        return m_mode_dims;
    }
    void set_dim(size_type n, size_type dim)
    {
        ASSERT(dim <= this->m_mode_capacity[n], "Failed to set mode dim larger than maximum size.");
        this->m_mode_dims[n]  = dim;
    }

    void conj()
    {
        this->as_matrix() = linalg::conj(this->as_matrix());
    }

    size_type nelems(bool use_max_dim = false) const{return this->hrank(use_max_dim)*this->dimen(use_max_dim);}
    size_type nset() const {return 1;}
    size_type max_hrank() const{return m_max_hrank;}
    size_type max_dimen() const{return m_max_dimen;}
    size_type max_dim(size_type n) const{return m_mode_capacity[n];}
    const std::vector<size_type>& max_dims() const{return m_mode_capacity;}

    linalg::reinterpreted_tensor<T, 1, backend> as_rank_1(bool use_max_dim = false)
    {
        if(!use_max_dim){return this->reinterpret_shape(dimen()*hrank());}
        else{return this->reinterpret_capacity(dimen(use_max_dim)*hrank(use_max_dim));}
    }

    linalg::reinterpreted_tensor<const T, 1, backend> as_rank_1(bool use_max_dim = false) const
    {
        if(!use_max_dim){return this->reinterpret_shape(dimen()*hrank());}
        else{return this->reinterpret_capacity(dimen(use_max_dim)*hrank(use_max_dim));}
    }
    
    linalg::reinterpreted_tensor<T, 2, backend> as_rank_2(bool use_max_dim = false)
    {
        if(!use_max_dim){return this->reinterpret_shape(dimen(), hrank());}
        else{return this->reinterpret_capacity(dimen(use_max_dim), hrank(use_max_dim));}
    }

    linalg::reinterpreted_tensor<const T, 2, backend> as_rank_2(bool use_max_dim = false) const
    {
        if(!use_max_dim){return this->reinterpret_shape(dimen(), hrank());}
        else{return this->reinterpret_capacity(dimen(use_max_dim), hrank(use_max_dim));}
    }

    linalg::reinterpreted_tensor<T, 3, backend> as_rank_3(size_type mode, bool use_max_dim = false)
    {
        try
        {
            ASSERT(mode <= nmodes(), "Failed to interpret ttn_node_data as rank 3 tensor.  The mode index is out of bounds.");
            if(mode < nmodes())
            {
                if(!use_max_dim)
                {
                    std::array<size_type, 3> shape{{1, dim(mode), hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
                    return this->reinterpret_shape(shape[0], shape[1], shape[2]);
                }
                else
                {
                    std::array<size_type, 3> shape{{1, max_dim(mode), max_hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=max_dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= max_dim(i);}
                    return this->reinterpret_capacity(shape[0], shape[1], shape[2]);
                }
            }
            else
            {
                if(!use_max_dim)
                {
                    return this->reinterpret_shape(dimen(), hrank(), 1);
                }
                else
                {
                    return this->reinterpret_capacity(max_dimen(), max_hrank(), 1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to reinterpret hierarchical tucker tensor node as a rank 3 tensor.");
        }
    }

    linalg::reinterpreted_tensor<const T, 3, backend> as_rank_3(size_type mode, bool use_max_dim = false) const
    {
        try
        {
            ASSERT(mode <= nmodes(), "Failed to interpret ttn_node_data as rank 3 tensor.  The mode index is out of bounds.");

            if(mode < nmodes())
            {
                if(!use_max_dim)
                {
                    std::array<size_type, 3> shape{{1, dim(mode), hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
                    return this->reinterpret_shape(shape[0], shape[1], shape[2]);
                }
                else
                {
                    std::array<size_type, 3> shape{{1, max_dim(mode), max_hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=max_dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= max_dim(i);}
                    return this->reinterpret_capacity(shape[0], shape[1], shape[2]);
                }        
            }
            else
            {
                if(!use_max_dim)
                {
                    return this->reinterpret_shape(dimen(), hrank(), 1);
                }
                else
                {
                    return this->reinterpret_capacity(max_dimen(), max_hrank(), 1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to reinterpret hierarchical tucker tensor node as a rank 3 tensor.");
        }
    }

    linalg::reinterpreted_tensor<T, 4, backend> as_rank_4(size_type mode)
    {
        ASSERT(mode < nmodes(), "Failed to interpret ttn_node_data as rank 3 tensor.  The mode index is out of bounds.");
        std::array<size_type, 4> shape{{1, dim(mode), 1, hrank()}};
        for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
        for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
        return this->reinterpret_shape(shape[0], shape[1], shape[2], shape[3]);
    }

    linalg::reinterpreted_tensor<const T, 4, backend> as_rank_4(size_type mode) const
    {
        ASSERT(mode < nmodes(), "Failed to interpret ttn_node_data as rank 3 tensor.  The mode index is out of bounds.");
        std::array<size_type, 4> shape{{1, dim(mode), 1, hrank()}};
        for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
        for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
        return this->reinterpret_shape(shape[0], shape[1], shape[2], shape[3]);
    }

    linalg::matrix<T, backend>& as_matrix(){return *this;}
    const linalg::matrix<T, backend>& as_matrix() const {return *this;} 

    void clear()
    {
        m_mode_dims.clear();
        matrix_type::clear();
    }

public:
    std::array<size_t, 3> expand_bond(size_type mode, size_type iadd, matrix_type& temp)
    {
        try
        {
            CALL_AND_HANDLE(temp.resize(this->shape(0), this->shape(1)), "Failed to resize temporary array.");
            CALL_AND_HANDLE(temp = this->as_matrix(), "Failed to copy array into temporary buffer.");

            auto atens = this->as_rank_3(mode);
            std::array<size_t, 3> _shape = atens.shape();
            if(mode == this->nmodes())
            {
                this->resize(this->hrank()+iadd, this->dims());
            }
            else
            {
                auto dims = this->dims();
                dims[mode] += iadd;
                this->resize(this->hrank(), dims);
            }
            this->as_matrix().fill_zeros();
            backend::rank_3_strided_copy(temp.buffer(), _shape[0], _shape[1], _shape[2], this->as_matrix().buffer(), _shape[1]+iadd);
            return _shape;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to expand tensor.");
        }
    }

    void expand_bond(size_type mode, size_type iadd, matrix_type& temp, std::mt19937& rng)
    {
        try
        {
            std::array<size_t, 3> _shape;
            CALL_AND_RETHROW(_shape = this->expand_bond(mode, iadd, temp));

            //now transform this object into the correct rank 3 tensor
            auto atens = this->as_rank_3(mode);
            temp.resize(this->shape(0), this->shape(1));
            auto ttens = temp.reinterpret_shape(atens.shape(1), atens.shape(0), atens.shape(2));
            
            CALL_AND_HANDLE(ttens = linalg::trans(atens, {1, 0, 2}), "Failed to transpose the atens object into utens.");
            auto tmat = ttens.reinterpret_shape(atens.shape(1), atens.shape(0)*atens.shape(2));

            using orthog = helper::orthogonal_vector<T, backend>;
            CALL_AND_RETHROW(orthog::pad_random_vectors(tmat, _shape[1], rng));

            //now transpose the results back into the original vector
            CALL_AND_HANDLE(atens = linalg::trans(ttens, {1, 0, 2}), "Failed to transpose the atens object into utens.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to expand tensor.");
        }
    }

    template <typename RType>
    void generate_random_orthogonal(size_type mode, matrix_type& temp, std::mt19937& rng, RType&& r)
    {
        auto atens = this->as_rank_3(mode);
        temp.resize(this->shape(0), this->shape(1));
        auto ttens = temp.reinterpret_shape(atens.shape(1), atens.shape(0), atens.shape(2));
        
        CALL_AND_HANDLE(ttens = linalg::trans(atens, {1, 0, 2}), "Failed to transpose the atens object into utens.");
        auto tmat = ttens.reinterpret_shape(atens.shape(1), atens.shape(0)*atens.shape(2));

        CALL_AND_HANDLE(r.resize(tmat.shape(1)), "Failed to resize random vector.");
        using orthog = helper::orthogonal_vector<T, backend>;
        CALL_AND_RETHROW(orthog::generate_random_vector(tmat, r, rng));
    }

    void expand_bond(size_type mode, size_type iadd, matrix_type& temp, const matrix_type& pad)
    {
        try
        {
            std::array<size_t, 3> _shape;
            CALL_AND_RETHROW(_shape = this->expand_bond(mode, iadd, temp));
            backend::rank_3_strided_append(pad.buffer(), _shape[0], _shape[1], _shape[2], iadd, this->as_matrix().buffer(), _shape[1]+iadd);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to expand tensor.");
        }
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<matrix<T, backend>>(this)), "Failed to serialise ttn_node_data object.  Error when serialising the base matrix object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dims", m_mode_dims)), "Failed to serialise ttn_node_object object.  Error when serialising mode dimensions.");
    }

    template <typename archive>
    void load(archive& ar) 
    {
        CALL_AND_HANDLE(ar(cereal::base_class<matrix<T, backend>>(this)), "Failed to serialise ttn_node_data object.  Error when serialising the base matrix object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dims", m_mode_dims)), "Failed to serialise ttn_node_object object.  Error when serialising mode dimensions.");
    }
#endif

public:
    void set_identity()
    {
        linalg::matrix<T, linalg::blas_backend> ch(this->size(0), this->size(1));
        for(size_type i=0; i<this->size(0); ++i){for(size_type j=0; j<this->size(1); ++j){ch(i, j) = (i == j ? 1.0 : 0.0);}}
        this->operator=(ch);
    }

    void set_random(std::mt19937& rng)
    {
        auto& mat = this->as_matrix();
        backend::fill_random_normal(mat.buffer(), mat.size(), rng);
    }

    void set_node_state(size_type i, std::mt19937& rng, bool random_unoccupied_initialisation=false)
    {
        size_type s1 = this->shape(0);
        size_type s2 = this->shape(1);

        linalg::matrix<T, linalg::blas_backend> ct(s2, s1);  ct.fill_zeros();
        ct(0, i) = T(1.0);

        using orthog = helper::orthogonal_vector<T, backend>;
        if(random_unoccupied_initialisation)
        {
            CALL_AND_RETHROW(orthog::pad_random_vectors(ct, 1, rng));
        }
        else
        {
            size_type cu = i;
            size_type cd = i;

            for(size_type j = 1; j < s2; ++j)
            {
                //for even numbers first attempt to add states with larger excitations and if we have reached the end add fewer excitations
                if (j%2 == 0)
                {
                    //
                    if(cu+1 != s1)
                    {
                        cu+=1;
                        ct(j, cu) = T(1.0);
                    }
                    else if(cd != 0)
                    {
                        cd-=1;
                        ct(j, cd) = T(1.0);
                    }
                    else{RAISE_EXCEPTION("Failed to set node state unable to set unoccupied vectors.");}
                }
                //for odd numbers attempt to add states with fewer excitations and if we have reached the end add states with more excitations
                else
                {
                    if(cd != 0)
                    {
                        cd-=1;
                        ct(j, cd) = T(1.0);
                    }
                    else if(cu+1 != s1)
                    {
                        cu+=1;
                        ct(j, cu) = T(1.0);
                    }
                    else{RAISE_EXCEPTION("Failed to set node state unable to set unoccupied vectors.");}
                }
            }

        }

        this->as_matrix() = linalg::trans(ct);
    }

    template <typename U, typename be> 
    void set_node_vector(const linalg::vector<U, be>& psi0, std::mt19937& rng)
    {

        size_type s1 = this->shape(0);
        size_type s2 = this->shape(1);

        linalg::matrix<T, linalg::blas_backend> ct(s2, s1);  ct.fill_zeros();
        CALL_AND_HANDLE(ct[0] = psi0, "Failed to assign psi0 in htucker.");
        using orthog = helper::orthogonal_vector<T, backend>;
        CALL_AND_RETHROW(orthog::pad_random_vectors(ct, 1, rng));

        this->as_matrix() = linalg::trans(ct);
    }

    void set_node_purification(std::mt19937& rng)
    {
        size_type s1 = this->shape(0);
        size_type s2 = this->shape(1);

        linalg::matrix<T, linalg::blas_backend> ct(s2, s1);  ct.fill_zeros();
        size_type ns = static_cast<size_type>(std::sqrt(s1));
        ASSERT(ns*ns == s1, "Cannot setup purification.  The node does not have square local dimension and consequently can not handle a composite system, ancilla state.");
        for(size_t i = 0; i < ns; ++i)
        {
            ct(0, i*ns + ((i + 1)%ns)) = 1.0/std::sqrt(ns);
        }
        using orthog = helper::orthogonal_vector<T, backend>;
        CALL_AND_RETHROW(orthog::pad_random_vectors(ct, 1, rng));
        this->as_matrix() = linalg::trans(ct);
    }

    static void query_node_sizes(const ttn_node_data<T, backend>& a, std::array<size_type, 3>& sizes, bool is_leaf = false)
    {
        size_type _size = a.size();            if(_size > sizes[0]){sizes[0] = _size;}
        size_type _capacity = a.capacity();    if(_capacity > sizes[1]){sizes[1] = _capacity;}

        size_type dim2i  = a.max_hrank()*a.max_hrank();   if(dim2i > sizes[2]){sizes[2] = dim2i;}

        //when querying the maximum capacity object for the U matrix we only need to test the nodes downwards
        //if we are at an interior node.  The leaf nodes never require decompositions pointing downwards. 
        if(!is_leaf)
        {
            for(size_type i = 0; i < a.nmodes(); ++i)
            {
                dim2i  = a.max_dim(i)*a.max_dim(i);   if(dim2i > sizes[2]){sizes[2] = dim2i;}
            }   
        }
    }

protected:
    template <typename Itype, typename Atype>
    typename std::enable_if<std::is_integral<Itype>::value, void>::type 
    setup_data_from_topology_node(const ntree_node<ntree<Itype, Atype>>& tree_iter, 
                                  const ntree_node<ntree<Itype, Atype>>& capacity_iter, 
                                  size_type ndims, 
                                  bool purification = false)                                  
    {

        this->m_mode_dims.resize(ndims);
        this->m_mode_capacity.resize(ndims);

        //first we need to go ahead and determine the size of the transfer or basis matrix corresponding to the current node.
        size_type size1 = tree_iter.value();
        size_type size2 = 1;

        //iterate over both the children of the node and the array storing the node info and set the
        //number of primitive functions in the node info array 
        size_type index2 = 0;
        for(auto& topology_child : tree_iter)
        {
            size2 *= topology_child.value();
            this->m_mode_dims[index2] = topology_child.value();
            ++index2;
        }

        size_type capacity1 = capacity_iter.value();
        size_type capacity2 = 1;
        index2=0;
        for(auto& capacity_child : capacity_iter)
        {
            capacity2 *= capacity_child.value();
            this->m_mode_capacity[index2] = capacity_child.value();
            ++index2;
        }

        //if we are at a leaf node and are a purification square the local hildbert space
        if(ndims == 1 && purification)
        {
            size_type mdim = this->m_mode_dims[0];
            size2 *= mdim;
            this->m_mode_dims[0] *= mdim;

            size_t cdim = this->m_mode_capacity[0];
            capacity2 *= cdim;
            this->m_mode_capacity[0] *= cdim;
        }

        this->m_max_hrank = capacity1;
        this->m_max_dimen = capacity2;

        ASSERT(size1 <= size2, "Failed to construct the ttn.  No single subtensor in the ttn can have a larger hierarchical rank than it has basis functions.");

        this->as_matrix().reallocate(capacity2*capacity1);
        this->as_matrix().resize(size2, size1);
        this->fill_zeros();
    }

    template <typename Itype, typename Atype>
    typename std::enable_if<std::is_integral<Itype>::value, void>::type 
    setup_data_from_topology_node(const ntree_node<ntree<std::vector<Itype>, Atype>>& tree_iter, 
                                  const ntree_node<ntree<std::vector<Itype>, Atype>>& capacity_iter, 
                                  size_type ndims, size_type ind, bool purification = false)
    {
        this->m_mode_dims.resize(ndims);
        this->m_mode_capacity.resize(ndims);

        //first we need to go ahead and determine the size of the transfer or basis matrix corresponding to the current node.
        size_type size1 = tree_iter.value()[ind];
        size_type size2 = 1;

        //iterate over both the children of the node and the array storing the node info and set the
        //number of primitive functions in the node info array 
        size_type index2 = 0;
        for(auto& topology_child : tree_iter)
        {
            size2 *= topology_child.value()[ind];
            this->m_mode_dims[index2] = topology_child.value()[ind];
            ++index2;
        }

        size_type capacity1 = capacity_iter.value()[ind];
        size_type capacity2 = 1;
        index2=0;
        for(auto& capacity_child : capacity_iter)
        {
            capacity2 *= capacity_child.value()[ind];
            this->m_mode_capacity[index2] = capacity_child.value()[ind];
            ++index2;
        }

        //if we are at a leaf node and are a purification square the local hildbert space
        if(ndims == 1 && purification)
        {
            size_type mdim = this->m_mode_dims[0];
            size2 *= mdim;
            this->m_mode_dims[0] *= mdim;

            size_t cdim = this->m_mode_capacity[0];
            capacity2 *= cdim;
            this->m_mode_capacity[0] *= cdim;
        }

        this->m_max_hrank = capacity1;
        this->m_max_dimen = capacity2;

        ASSERT(size1 <= size2, "Failed to construct the ttn.  No single subtensor in the ttn can have a larger hierarchical rank than it has basis functions.");

        this->as_matrix().reallocate(capacity2*capacity1);
        this->as_matrix().resize(size2, size1);
        this->fill_zeros();
    }
};

template <typename T, typename backend, typename = typename std::enable_if<std::is_same<backend, linalg::blas_backend>::value, void>::type> 
std::ostream& operator<<(std::ostream& os, const ttn_node_data<T, backend>& t)
{
    os << "dims: " << "[ ";     for(size_t i=0; i<t.nmodes(); ++i){os << t.dim(i) << (i+1 != t.nmodes() ? ", " : "]");}    os << std::endl;
    os << static_cast<const linalg::matrix<T, backend>&>(t) << std::endl;
    return os;
}

}

#include "node_traits/ttn_node_traits.hpp"
#include "node_traits/tensor_node_traits.hpp"
#include "../orthogonality/decomposition_engine.hpp"
#include "../orthogonality/root_to_leaf_decomposition.hpp"
#include "../orthogonality/leaf_to_root_decomposition.hpp"


namespace ttns
{

template <template <typename, typename> class node_data_type, typename T, typename backend>
class ttn_node_helper
{
public:
    using matrix_type = linalg::matrix<T, backend>;
    using value_type = node_data_type<T, backend>;
    using tree_type = tree_base<value_type>;
    using base_type = tree_node_base<tree_type>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using node_type = typename tree_type::node_type;
    using orthogonality_type = typename node_type::orthogonality_type;
    
    using engine_type = orthogonality::decomposition_engine<T, backend, false>;
    using r2l_core = orthogonality::root_to_leaf_decomposition_engine<T, backend>;
    using l2r_core = orthogonality::leaf_to_root_decomposition_engine<T, backend>;


public:
    static size_type contraction_capacity(const node_type& a, const node_type& b)
    {
        size_type mcap = 0;
        for(size_type i = 0; i < a.nset(); ++i)
        {
            size_type capacity = 1;
            for(size_type nm = 0; nm < a.dataview(i).nmodes(); ++nm)
            {
                capacity *= std::max(a.dataview(i).max_dim(nm), b.dataview(i).max_dim(nm));
            }
            if(capacity > mcap){mcap = capacity;}
        }
        return mcap;
    }

public:
    /*
     * Functions for handling subsections of the movement of the orthogonality centre up or down
     */
    static void decompose_down(node_type& A, orthogonality_type& orth, size_type mode, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false)
    {
        orth.resize_data(A);
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                r2l_core::evaluate(orth.eng(omp_get_thread_num()), A.dataview(i), orth.U(i), orth.R(i), orth.S(i), orth.work(i), mode, tol, nchi, true, orth.truncation_mode(), save_svd), 
                "Failed to compute the root to leaf decomposition for a node."
            );
        }
        orth.most_recent_node() = A.id();
    }

    static void decompose_up(node_type& A, orthogonality_type& orth, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false)
    {
        orth.resize_data(A);
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                l2r_core::evaluate(orth.eng(omp_get_thread_num()), A.dataview(i), orth.U(i), orth.R(i), orth.S(i), tol, nchi, true, orth.truncation_mode(), save_svd), 
                "Failed to evaluate the leaf_to_root_decomposition for a given node."
            );
        }
        orth.most_recent_node() = A.id();
    }

    /*
     * Function for applying results of orthogonality decomposition to a node
     */
    static void apply_to_node(node_type& A, orthogonality_type& orth)
    {       
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                r2l_core::apply_to_node(A.dataview(i), orth.U(i)),
                "Failed to apply result of decomposition to node."
            );
        }
    }
  
    /*
     * Functions for applying bond matrix.  Here we implement four functions that allow for us to apply bond matrices above this node
     * either to this node or its parent, or apply bond matrices below this node either to this node or its child
     */
    //apply the bond matrix stored in orth to the parent of this node.  
    static void apply_bond_matrix_to_parent(node_type& A, orthogonality_type& orth)
    {
        ASSERT(orth.most_recent_node() == A.id(), "Cannot apply bond matrix to parent.  The most recently decomposed node is not this node.");
        //apply the action of the current bond matrix to the node.  Here this applies the matrix along the bond pointing down into the node
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                l2r_core::apply_bond_matrix(A.dataview(i), A.parent().dataview(i), A.child_id(), orth.R(i), orth.work(i)),
                "Failed to apply bond matrix downwards along the bond."
            );
        }
    }

    //apply the bond matrix stored in orth from the parent of this node
    static void apply_bond_matrix_from_parent(node_type& A, orthogonality_type& orth)
    {
        ASSERT(orth.most_recent_node() == A.parent().id(), "Cannot apply bond matrix from parent.  The most recently decomposed node is not the parent of this node.");
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                r2l_core::apply_bond_matrix(A.dataview(i), orth.R(i), orth.work(i)),
                "Failed to apply bond matrix downwards along the bond."
            );
        }
    }

    static void apply_bond_matrix_to_child(node_type& A, orthogonality_type& orth, size_type mode)
    {
        ASSERT(orth.most_recent_node() == A.id(), "Cannot apply bond matrix to child.  The most recently decomposed node is not this node.");
        //apply the action of the current bond matrix to the node.  Here this applies the matrix along the bond pointing down into the node
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                r2l_core::apply_bond_matrix(A[mode].dataview(i), orth.R(i), orth.work(i)),
                "Failed to apply bond matrix downwards along the bond."
            );
        }
    }

    static void apply_bond_matrix_from_child(node_type& A, orthogonality_type& orth, size_type mode)
    {
        ASSERT(orth.most_recent_node() == A[mode].id(), "Cannot apply bond matrix from child.  The most recently decomposed node is not the expected child of this node.");
        //apply the action of the current bond matrix to the node.  Here this applies the matrix along the bond pointing down into the node
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                l2r_core::apply_bond_matrix(A[mode].dataview(i), A.dataview(i), mode, orth.R(i), orth.work(i)),
                "Failed to apply bond matrix downwards along the bond."
            );
        }
    }

    /*
     * Functions for handling the movement of the orthogonality centre up or down
     */
    static void shift_orthogonality_down(node_type& A, orthogonality_type& orth, size_type mode, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false)
    {
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                r2l_core::evaluate(orth.eng(omp_get_thread_num()), A.dataview(i), orth.U(i), orth.R(i), orth.S(i), orth.work(i), mode, tol, nchi, true, orth.truncation_mode(), save_svd), 
                "Failed to compute the root to leaf decomposition for a node."
            );

            //now we need to apply to node and apply the bond matrix enforcing truncation
            CALL_AND_HANDLE(
                r2l_core::apply_with_truncation(A.dataview(i), A[mode].dataview(i), mode, orth.R(i), orth.U(i)), 
                "Failed to apply the result of the root to leaf decomposition imposing the correct truncation."
            );
        }
        orth.most_recent_node() = A.id();
    }

    static void shift_orthogonality_up(node_type& A, orthogonality_type& orth, real_type tol = real_type(0), size_type nchi = 0, bool save_svd = false)
    {
#ifdef USE_OPENMP
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for num_threads(orth.nthreads()) default(shared) if(orth.parallelise() && A.nset() > 1)
#endif
#endif
        for(size_type i = 0; i < A.nset(); ++i)  
        {
            CALL_AND_HANDLE(
                l2r_core::evaluate(orth.eng(omp_get_thread_num()), A.dataview(i), orth.U(i), orth.R(i), orth.S(i), tol, nchi, true, orth.truncation_mode(), save_svd), 
                "Failed to evaluate the leaf_to_root_decomposition for a given node."
            );

            //now we need to apply to node and to parent making use of truncation
            CALL_AND_HANDLE(
                l2r_core::apply_with_truncation(A.dataview(i), A.parent().dataview(i), A.child_id(), orth.R(i), orth.U(i)), 
                "Failed when applying the result of the leaf_to_root_decomposition imposing the correct truncation."
            );
        }
        orth.most_recent_node() = A.id();
    }

};


template <typename T, typename backend> 
class tree_node<tree_base<ttn_node_data<T, backend> > > : 
    public tree_node_base<tree_base<ttn_node_data<T, backend> > >
{
    static_assert(std::is_base_of<linalg::backend_base, backend>::value, "The second template argument to the ttn_node object must be a valid backend.");
public:
    using matrix_type = linalg::matrix<T, backend>;
    using value_type = ttn_node_data<T, backend>;
    using tree_type = tree_base<value_type>;
    using base_type = tree_node_base<tree_type>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using node_type = typename tree_type::node_type;
    using self_type = node_type;
    using node_helper = ttn_node_helper<ttn_node_data, T, backend>;

    using bond_matrix_type = matrix_type;
    using population_matrix_type = linalg::diagonal_matrix<real_type, backend>;

    using engine_type = orthogonality::decomposition_engine<T, backend, false>;
    using hrank_type = size_type;

    friend class tree<value_type>;
    friend class tree_base<value_type>;
    friend ttn<T, backend>;
    template <template <typename, typename> class nt, typename U, typename be> friend class ttn_base;

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
            size_type maxcapacity = m_maxcapacity;
            size_type maxsize = m_maxsize;
            m_workspace.reallocate(maxcapacity);
            m_U.reallocate(maxcapacity);
            m_U.resize(1, maxsize);

            for(const auto& a : nodes)
            {
                CALL_AND_HANDLE(r2l_core::resize_r_matrix(a(), m_R, true), "Failed to resize elements of the r tensor.");
                m_S.resize(m_R.shape(0), m_R.shape(1));
            }
            try
            {
                m_ortho_engine.template resize<r2l_core>(nodes, m_U, m_R, true);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to resize the decomposition engine object.");
            }
            m_initialised = true;
        }

        void resize_data(const self_type& a)
        {
            m_U.resize(a().shape(0), a().shape(1));
            m_workspace.resize(a().shape(0), a().shape(1));
        }
        void clear()
        {
            m_ortho_engine.clear();
            m_U.clear();
            m_R.clear();
            m_S.clear();
            m_workspace.clear();
            m_most_recent_node = 0;
            m_maxsize = 0;
            m_maxcapacity = 0;
            m_initialised = false;
        }

        template <typename tree>
        void query_sizes(tree& nodes)
        {
            std::array<size_type, 3> sizes{{0,0,0}};
            for(const auto& a : nodes){ttn_node_data<T, backend>::query_node_sizes(a(), sizes, a.is_leaf());}
            m_maxsize = sizes[0];
            m_maxcapacity = (sizes[1] > sizes[2] ? sizes[1] : sizes[2]);
        }


        orthogonality::truncation_mode& truncation_mode(){return m_truncation_mode;}
        const orthogonality::truncation_mode& truncation_mode() const{return m_truncation_mode;}

        size_type most_recent_node() const{return m_most_recent_node;}
        size_type& most_recent_node(){return m_most_recent_node;}

        linalg::diagonal_matrix<real_type, backend>& population_matrix(){return m_S;}
        matrix_type& bond_matrix(){return m_R;}
        engine_type& eng(){return m_ortho_engine;}
        matrix_type& work(){return m_workspace;}
        matrix_type& R(){return m_R;}
        matrix_type& U(){return m_U;}
        linalg::diagonal_matrix<real_type, backend>& S(){return m_S;}

        const linalg::diagonal_matrix<real_type, backend>& population_matrix() const{return m_S;}
        const matrix_type& bond_matrix() const{return m_R;}
        const engine_type& eng() const{return m_ortho_engine;}
        const matrix_type& work() const{return m_workspace;}
        const matrix_type& R() const{return m_R;}
        const matrix_type& U() const{return m_U;}
        const linalg::diagonal_matrix<real_type, backend>& S() const{return m_S;}

        linalg::diagonal_matrix<real_type, backend>& population_matrix(size_type ){return m_S;}
        matrix_type& bond_matrix(size_type ){return m_R;}
        engine_type& eng(size_type i)
        {       
            ASSERT(i == 0, "Cannot access engine object index out of bounds.");
            return m_ortho_engine;
        }
        matrix_type& work(size_type ){return m_workspace;}
        matrix_type& R(size_type ){return m_R;}
        matrix_type& U(size_type ){return m_U;}
        linalg::diagonal_matrix<real_type, backend>& S(size_type ){return m_S;}

        const linalg::diagonal_matrix<real_type, backend>& population_matrix(size_type ) const{return m_S;}
        const matrix_type& bond_matrix(size_type ) const{return m_R;}
        const engine_type& eng(size_type i) const
        {   
            ASSERT(i == 0, "Cannot access engine object index out of bounds.");
            return m_ortho_engine;
        }
        const matrix_type& work(size_type ) const{return m_workspace;}
        const matrix_type& R(size_type ) const{return m_R;}
        const matrix_type& U(size_type ) const{return m_U;}
        const linalg::diagonal_matrix<real_type, backend>& S(size_type ) const{return m_S;}
        
        size_type nthreads() const{return 1;}
        bool parallelise() const{return false;}
    protected:
        engine_type m_ortho_engine;
        matrix_type m_U;
        matrix_type m_R;
        linalg::diagonal_matrix<real_type, backend> m_S;
        matrix_type m_workspace;
        size_type m_most_recent_node = 0;
        size_type m_maxsize = 0;
        size_type m_maxcapacity = 0;
        bool m_initialised = false;

        orthogonality::truncation_mode m_truncation_mode = orthogonality::truncation_mode::singular_values_truncation;
    };

protected:
    using base_type::m_data;
    using base_type::m_children;
    using base_type::m_parent;

public:
    std::ostream& output_bond_dimensions(std::ostream& os)
    {
        os << "(";
        os << hrank();
        for(size_t i=0; i<this->size(); ++i)
        {
            this->operator[](i).output_bond_dimensions(os);
        }
        os << ")";
        return os;
    }


public:
    tree_node() : base_type(){}

    void zero(){m_data.fill_zeros();}

    size_type maxsize() const{return m_data.size();}
    size_type maxcapacity() const{return m_data.capacity();}
    size_type buffer_maxcapacity() const{return m_data.capacity();}
    size_type maxhrank(bool use_capacity = false) const{return m_data.hrank(use_capacity);}

    void get_hrank(size_type& res ) const{res = m_data.hrank();}
    size_type hrank() const{return m_data.hrank();}
    size_type nmodes() const{return m_data.nmodes();}
    size_type dimen() const {return m_data.dimen();}
    size_type nset() const {return 1;}
    size_type dim(size_type n) const{return m_data.dim(n);}
    size_type max_dim(size_type n) const{return m_data.max_dim(n);}
    const std::vector<size_type>& dims() const{return m_data.dims();}
    size_type nbonds() const{return this->size() + (this->is_root() ? 0 : 1);}
    size_t buffer_size() const
    {
        return m_data.size();
    }

    const ttn_node_data<T, backend>& operator()(size_type i1) const {ASSERT(i1 == 0, "Index out of bounds."); return m_data;}
    ttn_node_data<T, backend>& operator()(size_type i1) {ASSERT(i1 == 0, "Index out of bounds."); return m_data;}
    const value_type& operator()() const {return m_data;}
    value_type& operator()() {return m_data;}

    const ttn_node_data<T, backend>& dataview(size_type) const{return m_data;}
    ttn_node_data<T, backend>& dataview(size_type){return m_data;}

#ifdef CEREAL_LIBRARY_FOUND
    friend class serialisation_node_save_wrapper<node_type, size_type>;
    friend class serialisation_node_load_wrapper<node_type, size_type>;
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<tree_node_base<tree_base<ttn_node_data<T, backend> > >>(this)), "Failed to serialise ttn_node object.  Error when serialising the base object.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<tree_node_base<tree_base<ttn_node_data<T, backend> > >>(this)), "Failed to serialise ttn_node object.  Error when serialising the base object.");
    }
#endif

public:
    template <typename U, typename be>
    bool can_fit_node(const ttn_node_data<U, be>& o) const{return m_data.can_fit_node(o);}

public:
    real_type norm() const{auto vec = m_data.as_rank_1();    return std::sqrt(std::real(linalg::dot_product(linalg::conj(vec), vec))); }

    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, self_type&>::type operator*=(const U& u){m_data.as_matrix() *= u;    return *this;    }

    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, self_type&>::type operator/=(const U& u){m_data.as_matrix() /= u;    return *this;    }

public:
    template <typename Itype, typename Atype>
    typename std::enable_if<std::is_integral<Itype>::value, void>::type 
    setup_data_from_topology_node(const ntree_node<ntree<Itype, Atype>>& tree_iter, 
                                  const ntree_node<ntree<Itype, Atype>>& capacity_iter, 
                                  size_type ndims,
                                  size_type /* nset */ = 1,
                                  bool purification = false)
    {
        CALL_AND_RETHROW(m_data.setup_data_from_topology_node(tree_iter, capacity_iter, ndims, purification));
    }

public:
    void set_node_identity(size_type /*  set_index */ = 0){m_data.set_identity();}
    void set_node_random(std::mt19937& rng){m_data.set_random(rng);}
    void set_node_random(size_type /* set_index */, std::mt19937& rng){m_data.set_random(rng);}

    void set_leaf_node_state(size_type /* set_index */, size_type i, std::mt19937& rng, bool random_unoccupied_initialisation=false)
    {
        ASSERT(this->is_leaf(), "Function is only applicable for leaf state nodes.");
        this->m_data.set_node_state(i, rng, random_unoccupied_initialisation);
    }

    template <typename U, typename be> 
    void set_leaf_node_vector(size_type /* set_index */, const linalg::vector<U, be>& psi0, std::mt19937& rng)
    {
        ASSERT(this->is_leaf(), "Function is only applicable for leaf state nodes.");
        this->m_data.set_node_vector(psi0, rng);
    }

    void set_leaf_purification(size_type /* set_index */, std::mt19937& rng)
    {
        ASSERT(this->is_leaf(), "Function is only applicable for leaf state nodes.");
        this->m_data.set_node_purification(rng);
    }

    void conj()
    {
        this->m_data.conj();
    }
public:
    static size_type contraction_capacity(const node_type& a, const node_type& b){CALL_AND_RETHROW(return node_helper::contraction_capacity(a, b));}

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


