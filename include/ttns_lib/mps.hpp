#ifndef TTNS_LIB_MPS_HPP
#define TTNS_LIB_MPS_HPP

#include <random>

#include "mps_nodes/httensor_node.hpp"
#include <common/tmp_funcs.hpp>

#include "algorithms/core/decomposition_engine.hpp"
#include "algorithms/core/root_to_leaf_decomposition.hpp"
#include "algorithms/core/leaf_to_root_decomposition.hpp"

namespace ttns
{

template <typename T, typename backend = blas_backend>
class mps 
{
public:
    static_assert(is_number<T>::value, "The first template argument to the mps object must be a valid number type.");
    static_assert(guarenteed_valid_backend<backend>::value, "The second template argument to the mps object must be a valid backend.");

    using real_type = typename linalg::get_real_type<T>::type;

    using matrix_type = matrix<T, backend>;
    using base_type = tree<httensor_node_data<T, backend> >;

    using value_type = matrix_type;
    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using size_type = typename backend::size_type;

    using engine_type = decomposition_engine<T, backend, false>;
private:
    std::vector<linalg::tensor<T, backend, 3>> m_nodes;

    linalg::matrix<T> m_active_bond_matrix;
    linalg::matrix<T> m_U;
    linalg::matrix<T> m_workspace;

    engine_type m_ortho_engine;

    size_type m_orthogonality_centre = 0;
    bool m_has_orthogonality_centre = false;
    bool m_orthogonalisation_initialised = false;
    size_type m_maxsize;
    size_type m_maxcapacity;
    bool m_guarenteed_valid = false;
public:
    using base_type::size;

public:
    mps() {}
    mps(size_type N) : m_nodes(N) {}
    mps(size_type N, size_type d, size_type chi)
    {
        CALL_AND_HANDLE(resize(N, d, chi), "Failed to construct mps object.");
    }

    template <typename U, typename be>
    mps(const mps<U, be>& other)
    {
        this->operator=(other);
    }

    template <typename dtype, typename chitype>
    mps(const dtype& d, const chitype& chi)
    {
        CALL_AND_HANDLE(resize(d, chi), "Failed to construct mps object.");
    }

    template <typename U, typename be>
    mps& operator=(const mps<U, be>& other)
    {
        m_nodes = other.m_nodes;
        m_active_bond_matrix = other.m_active_bond_matrix;
        m_U = other.m_U;
        m_workspace = other.m_workspace;

        m_ortho_engine = other.m_ortho_engine;
        m_orthogonality_centre = other.m_orthogonality_centre;

        m_orthogonality_centre = other.m_orthogonality_centre;
        m_has_orthogonality_centre = other.m_has_orthogonality_centre;
        m_orthogonality_initialised = other.m_orthogonality_initialised;

        m_guaranteed_valid = other.m_guaranteed_valid;
        return *this;
    }

    void resize(size_type N, size_type d, size_type chi)
    {
        CALL_AND_HANDLE(clear(), "Failed to resize mps object.  Failed to clear currently allocated data.");
        try
        {
            if(N > 0)
            {
                m_nodes.resize(N);
                if(N == 1)
                {
                    m_nodes[0].resize(1, d, 1);
                }
                else
                {
                    m_nodes[0].resize(1, d, chi);
                    for(size_type i = 1; i < N-1; ++i){m_nodes[i].resize(chi, d, chi);}
                    m_nodes[N-1].resize(chi, d, 1);

                }
            }
            m_has_orthogonality_centre = false;
            m_guarenteed_valid = true;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize MPS.");
        }
    }

    template <typename dtype, typename chitype>
    void resize(const dtype& d, const chitype& chi)
    {
        CALL_AND_HANDLE(clear(), "Failed to resize mps object.  Failed to clear currently allocated data.");
        try
        {
            size_type N = d.size();
            ASSERT(chi.size() + 1 == N, "Failed to resize mps, bond dimension array and local hilbert space dimension arrays do not have compatible sizes.");

            if(N > 0)
            {
                m_nodes.resize(N);
                if(N == 1)
                {
                    m_nodes[0].resize(1, d[0], 1);
                }
                else
                {
                    m_nodes[0].resize(1, d[0], chi[0]);
                    for(size_type i = 1; i < N-1; ++i){m_nodes[i].resize(chi[i-1], d[i], chi[i]);}
                    m_nodes[N-1].resize(chi[N-2], d[N-1], 1);

                }
            }
            m_has_orthogonality_centre = false;
            m_guarenteed_valid = true;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize MPS.");
        }
    }

    template <typename rng>
    void random(rng& _rng)
    {
        try
        {
            if(!m_orthogonalisation_initialised)
            {
                setup_orthog();
            }

            std::normal_distribution<real_type> dist(0, 1);
            for(auto& n : utils::reverse(m_nodes))
            {
                backend::fill_tensor_random_normal(n().as_matrix());

            }
            m_orthogonality_centre = 0;
            m_has_orthogonality_centre = true;
            for(auto& ch : m_nodes){ch.set_is_orthogonalised(true);}
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise random tensors.");
        }
    }

    void clear()
    {
        try
        {
            m_nodes.clear();
            m_ortho_engine.clear();
            m_U.clear();
            m_active_bond_matrix.clear();
            m_orthogonality_centre = 0;
            m_has_orthogonality_centre = false;

            m_orthogonalisation_initialised = false;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear mps object.");
        }
    }


    void zero(){for(auto& ch : m_nodes){ch.fill_zeros();} m_has_orthogonality_centre = false;}

    const std::vector<size_type>& mode_dimensions() const{return m_dim_sizes;}
    size_type dim(size_type i) const
    {
        ASSERT(i < m_nodes[i].size(), "Index out of bounds.");
        return m_nodes[i].shape(1);
    }
    size_type chi(size_type i) const
    {
        ASSERT(i+1 < m_nodes[i].size(), "Index out of bounds.");
        return m_nodes[i].shape(2);
    }

    size_type nbonds() const noexcept {return m_nodes.size()-1;}
    size_type nmodes() const noexcept {return m_nodes.size();}
    size_type ntensors() const noexcept{return m_nodes.size();}
    size_type max_size() const noexcept {return m_maxsize;}
    size_type max_capacity() const noexcept {return m_maxcapacity;}
    size_type orthogonality_centre() const noexcept{return m_orthogonality_centre;}

    //functions for accessing the underlying tensors in the object
    const linalg::tensor<T, backend, 3>& operator[](size_type i) const
    {
        return m_nodes[i];
    }
    linalg::tensor<T, backend, 3>& operator[](size_type i)
    {
        if(i != m_orthogonality_centre){m_has_orthogonality_centre = false;}
        m_guaranteed_valid = false;
        return m_nodes[i];
    }

    const linalg::tensor<T, backend, 3>& at(size_type i) const
    {
        ASSERT(i < m_nodes.size(), "Index out of bounds.");
        return m_nodes[i];
    }
    linalg::tensor<T, backend, 3>& at(size_type i)
    {
        ASSERT(i < m_nodes.size(), "Index out of bounds.");
        if(i != m_orthogonality_centre){m_has_orthogonality_centre = false;}
        m_guaranteed_valid = false;
        return m_nodes[i];
    }
    const linalg::tensor<T, backend, 3>& operator()(size_type i) const
    {
        ASSERT(i < m_nodes.size(), "Index out of bounds.");
        return m_nodes[i];
    }
    linalg::tensor<T, backend, 3>& operator()(size_type i)
    {
        ASSERT(i < m_nodes.size(), "Index out of bounds.");
        if(i != m_orthogonality_centre){m_has_orthogonality_centre = false;}
        m_guaranteed_valid = false;
        return m_nodes[i];
    }

    size_type nelems() const 
    {
        if(m_nodes.size() == 0){return 0;}
        size_type nelems = 1;
        for(size_type i=0; i<m_dim_sizes.size(); ++i){nelems *= m_nodes[i].size(1);}
        return nelems;
    }

    bool check_valid() const
    {
        if(!m_guaranteed_valid)
        {
            //first check that the bond dimensions inside the tensor are sensible
            bool is_valid = (m_nodes[0].shape(0) == 1 && m_nodes[m_nodes.size()-1].shape(2) == 1);
            for(size_t i = 0; m_nodes.size()-1; ++i)
            {
                is_valid = is_valid && (m_nodes[i].shape(2) == m_nodes[i+1].shape[0]);
            }

        }
        return m_guaranteed_valid;
    }

    //a function for ensuring that the ttns object is valid.  This will start by checking 
    void sanitise()
    {
        if(!m_guaranteed_valid)
        {
            
        }
    }

    bool has_orthogonality_centre() const{return m_has_orthogonality_centre;}
    void force_set_orthogonality_centre(size_t index)
    {
        ASSERT(index < m_nodes.size(), "Failed to set orthogonality centre. Index out of bounds.");
        m_orthogonality_centre = index; m_has_orthogonality_centre = true;
    }

    bool guaranteed_valid() const{return m_guaranteed_valid;}
    void force_guaranteed_valid(){m_guaranteed_valid = true;}

    //function for shifting the orthogonality centre along a given bond of the current orthogonality centre.  Potentially with truncation if either the tol variable or the nchi variables are set
    //and are less than the current dimension of this bond
    void shift_orthogonality_centre(size_t bond_index, real_type tol = real_type(0), size_type nchi = 0)
    {
        try
        {
            if(!m_orthogonalisation_initialised)
            {
                setup_orthog();
            }

            ASSERT(m_has_orthogonality_centre, "The orthogonality centre must be specified in order to allow for it to be shifted.");

            if(m_orthogonality_centre == 0 || m_orthogonality_centre + 1 = m_nodes.size())
            {
                ASSERT(bond_index < 1, "Failed to shift orthogonality centre along bond.  Bond index out of bounds.");
            }
            else
            {
                ASSERT(bond_index < 2, "Failed to shift orthogonality centre along bond.  Bond index out of bounds.");
            }

        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to shift orthogonality centre.");
        }
    }


    void set_orthogonality_centre(size_type index)
    {
        try
        {
            ASSERT(index < m_nodes.size(), "Failed to set orthogonality centre. Index out of bounds.");

            if(!m_orthogonalisation_initialised)
            {
                setup_orthog();
            }

            if(!m_has_orthogonality_centre)
            {
                if(index < (m_nodes.size()-1)/2)
                {
                    CALL_AND_HANDLE(_right_orthogonalise(), "Failed to orthogonalise non-orthogonal TTN object.");
                }
                else
                {
                    CALL_AND_HANDLE(_left_orthogonalise(), "Failed to orthogonalise non-orthogonal TTN object.");
                }
            }
            CALL_AND_HANDLE(_set_orthogonality_centre(index), "Failed to handle TTN with orthogonality centre.  Failed to shift node to index.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to set orthogonality centre for mps.");
        }
    }


    //takes a generic TTN and shifts the orthogonality centre to the root node
    void right_orthogonalise()
    {
        try
        {
            if(!m_orthogonalisation_initialised)
            {
                setup_orthog();
            }

            if(m_has_orthogonality_centre)
            {
                CALL_AND_HANDLE(_set_orthogonality_centre(0), "Failed to handle TTN with orthogonality centre.  Failed to shift node to root.");
            }
            else
            {
                //here we need to perform the leaf to root decomposition.
                CALL_AND_HANDLE(_right_orthogonalise(), "Failed to orthogonalise non-orthogonal TTN object.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to orthogonalise general mps.");
        }
    }

    //takes a generic TTN and shifts the orthogonality centre to the root node
    void left_orthogonalise()
    {
        try
        {
            if(!m_orthogonalisation_initialised)
            {
                setup_orthog();
            }

            if(m_has_orthogonality_centre)
            {
                CALL_AND_HANDLE(_set_orthogonality_centre(m_nodes.size()-1), "Failed to handle TTN with orthogonality centre.  Failed to shift node to root.");
            }
            else
            {
                //here we need to perform the leaf to root decomposition.
                CALL_AND_HANDLE(_left_orthogonalise(), "Failed to orthogonalise non-orthogonal TTN object.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to orthogonalise general mps.");
        }
    }

    size_type get_leaf_index(size_type lid)
    {
        ASSERT(lid < m_nleaves, "Invalid leaf index.");
        return m_leaf_indices[lid];
    }

public:

protected:
    void _left_orthogonalise()
    {
        try
        {
            using common::rzip;
            //check whether A is already orthogonalised.  If it is we don't need to do anything.
            for(auto& a : m_nodes)
            {
                m_U.resize(a().shape(0), a().shape(1));
                m_workspace.resize(a().shape(0), a().shape(1));

            }
            this->force_set_orthogonality_centre(m_nodes.size()-1);
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

    void _right_orthogonalise()
    {
        try
        {
            using common::rzip;
            //check whether A is already orthogonalised.  If it is we don't need to do anything.
            for(auto& a : utils::reverse(m_nodes))
            {
                m_U.resize(a().shape(0), a().shape(1));
                m_workspace.resize(a().shape(0), a().shape(1));
            }
            this->force_set_orthogonality_centre(m_nodes.size()-1);
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

    void _set_orthogonality_centre(size_type index)
    {
        try
        {
            while(m_orthogonality_centre < index)
            {
                CALL_AND_HANDLE(this->shift_orthogonality_centre(0), "Failed to shift orthogonality centre left.");
            }

            while(m_orthogonality_centre > index)
            {
                CALL_AND_HANDLE(this->shift_orthogonality_centre(1), "Failed to shift orthogonality centre right.");
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
    void query_sizes()
    {
        size_type maxsize = 0;
        size_type maxcapacity = 0;
        size_type max_cdim2 = 0;
        size_type max_dim2 = 0;

        for(const auto& a : m_nodes)
        {
            size_type _size = a().size();            if(_size > maxsize){maxsize = _size;}
            size_type _capacity = a().capacity();    if(_capacity > maxcapacity){maxcapacity = _capacity;}

            size_type dim2i  = a().max_hrank()*a().max_hrank();   if(dim2i > max_cdim2){max_cdim2 = dim2i;}
            dim2i  = a().hrank()*a().hrank();   if(dim2i > max_cdim2){max_cdim2 = dim2i;}
            for(size_type i = 0; i < a.nmodes(); ++i)
            {
                dim2i  = a().max_dim(i)*a().max_dim(i);   if(dim2i > max_cdim2){max_cdim2 = dim2i;}
                dim2i  = a().dim(i)*a().dim(i);   if(dim2i > max_dim2){max_dim2 = dim2i;}
            }
        }
        if(max_cdim2 > maxcapacity){maxcapacity = max_cdim2;}

        m_maxsize = maxsize;
        m_maxcapacity = maxcapacity;
    }


    void setup_orthog()
    {

        m_workspace.reallocate(m_maxcapacity);
        m_U.reallocate(m_maxcapacity);
        m_U.resize(1, m_maxsize);

        for(const auto& a : m_nodes)
        {
            CALL_AND_HANDLE(r2l_core::resize_r_matrix(a(), m_active_bond_matrix, true), "Failed to resize elements of the r tensor.");
        }
        try
        {
            m_ortho_engine.template resize<r2l_core>(*this, m_U, m_active_bond_matrix, true);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize the decomposition engine object.");
        }
        m_orthogonalisation_initialised =true;
    }


#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("nodes", m_nodes)), "Failed to serialise mps object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimensions", m_dim_sizes)), "Failed to serialise mps object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_indices", m_leaf_indices)), "Failed to serialise mps object.  Failed to serialise its leaf_indices.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("orthogonality_centre", m_orthogonality_centre)), "Failed to seriesalise mps object. Failed to serialise orthogonality centre.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("has_orthogonality_centre", m_has_orthogonality_centre)), "Failed to seriesalise mps object. Failed to serialise orthogonality centre.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("nodes", m_nodes)), "Failed to serialise mps object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimensions", m_dim_sizes)), "Failed to serialise mps object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_indices", m_leaf_indices)), "Failed to serialise mps object.  Failed to serialise its leaf_indices.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("orthogonality_centre", m_orthogonality_centre)), "Failed to seriesalise mps object. Failed to serialise orthogonality centre.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("has_orthogonality_centre", m_has_orthogonality_centre)), "Failed to seriesalise mps object. Failed to serialise orthogonality centre.");
        query_sizes();
    }
#endif

};

template <typename T, typename backend>
using mps_node = typename linalg::tensor<T, backend, 3>;

template <typename T, typename backend> 
std::ostream& operator<<(std::ostream& os, const mps<T, backend>& t)
{
    return os;
}


}   //namespace ttns

#endif  // TTNS_LIB_MPS_HPP //

