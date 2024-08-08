#ifndef TTNS_LIB_TTN_ORTHOGONALITY_DECOMPOSITION_ENGINE_HPP
#define TTNS_LIB_TTN_ORTHOGONALITY_DECOMPOSITION_ENGINE_HPP

#include <common/exception_handling.hpp>
#include <common/zip.hpp>
#include <linalg/decompositions/singular_value_decomposition/singular_value_decomposition.hpp>

namespace ttns
{
namespace orthogonality
{

enum truncation_mode
{
    weight_truncation = 0,
    singular_values_truncation = 1
};


//TODO: Need to fix the code so that it can handle weird the case of (1(7(7))(1(X))). 
template <typename T, typename backend=blas_backend, bool use_divide_and_conquer = true>
class decomposition_engine
{
public:
    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename backend::size_type;
    using matrix_type = matrix<T, backend>;
    using dmat_type = diagonal_matrix<real_type, backend>;
    using dmat_host_type = diagonal_matrix<real_type>;
    using svd_engine = singular_value_decomposition<matrix_type, use_divide_and_conquer>;
public:
    decomposition_engine() {}
    decomposition_engine(const decomposition_engine& o) = default;
    decomposition_engine(decomposition_engine&& o) = default;

    decomposition_engine& operator=(const decomposition_engine& o) = default;
    decomposition_engine& operator=(decomposition_engine&& o) = default;

    template <typename decomp_type, typename resize_obj, typename Utype, typename Rtype>
    void resize(const resize_obj& A, Utype& U, Rtype& r, bool use_capacity = false)
    {
        try
        {
            CALL_AND_HANDLE(resize_buffers<decomp_type>(A, use_capacity), "Failed to resize the buffers.");

            size_type max_ws = 0;
            //iterate over all nodes and determine the maximum worksize
            for(auto& a : A)
            {
                size_type ws;
                CALL_AND_HANDLE(ws = decomp_type::maximum_work_size_node(*this, a(), U, r, use_capacity), "Failed to when attempting to query maximum work size for node.");
                if(ws > max_ws){max_ws = ws;}
            }
            CALL_AND_HANDLE(m_svd.resize_work_space(max_ws), "Faile to resize work space for svd object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize decomposition engine object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_s.clear(), "Failed to clear the s matrix.");
            CALL_AND_HANDLE(m_temp.clear(), "Failed to clear a temporary working matrix.");
            CALL_AND_HANDLE(m_temp2.clear(), "Failed to clear a temporary working matrix..");
            CALL_AND_HANDLE(m_svd.clear(), "Failed to clear the svd engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the decomposition engine object.");
        }
    }

    template <typename Atype, typename Utype, typename Vtype>
    size_type query_work_size(const Atype& A, Utype& U, Vtype& V)
    {
        try
        {
            CALL_AND_HANDLE(m_s.resize(A.shape()), "Failed to resize m_s matrix.");
            size_type ws;
            CALL_AND_HANDLE(ws = m_svd.query_work_space(A, m_s, U, V, false), "Failed when making query work space call for the underlying svd object.");
            return ws;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed when querying the work size for the decomposition engine object.");
        }
    }


    template <typename Atype>
    void operator()(const Atype& A)
    {
        try
        {
            //check that the temporary arrays have the correct capacity
            ASSERT(A.shape(1) <= m_s.capacity(), "The matrix of singular values does not have sufficient capacity.");

            CALL_AND_HANDLE(m_s.resize(A.shape(1), A.shape(1)), "Failed to resize S matrix.");
            CALL_AND_HANDLE(m_svd(A, m_s), "Failed when evaluating the decomposition.")
            CALL_AND_HANDLE(m_shost = m_s, "Failed to copy singular values to host.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying the decomposition engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the decomposition engine.");
        }
    }


    template <typename Atype, typename Utype, typename Vtype>
    size_type operator()(const Atype& A, Utype& U, Vtype& V, real_type tol = real_type(0), size_type nchi = 0, bool rel_truncate = false, truncation_mode trunc_mode = truncation_mode::singular_values_truncation, bool save_shost = false)
    {
        try
        {
            //check that the temporary arrays have the correct capacity
            ASSERT(V.size() <= m_temp.capacity(), "The temporary matrix does not have sufficient capacity.");
            ASSERT(A.shape(1) <= m_s.capacity(), "The matrix of singular values does not have sufficient capacity.");

            CALL_AND_HANDLE(m_s.resize(A.shape(1), A.shape(1)), "Failed to resize S matrix.");
            CALL_AND_HANDLE(m_temp.resize(V.shape()), "Failed to resize temporary V matrix to ensure it has the correct shape.");
            CALL_AND_HANDLE(m_svd(A, m_s, U, m_temp, A.shape(0) < A.shape(1)), "Failed when evaluating the decomposition.")

            //determine the bond dimension to retain given the user specified tol and nchi arrays
            size_type bond_dimension = m_s.size();
            if(nchi > 0)
            {
                bond_dimension = std::min(nchi, bond_dimension);
            }
            real_type snorm = 0.0;
            if(tol > real_type(0))
            {   
                CALL_AND_HANDLE(m_shost = m_s, "Failed to copy singular values to host.");
                size_type nb = 0;

                if(rel_truncate)
                {
                    for(size_type i = 0; i < bond_dimension; ++i)
                    {
                        snorm += m_shost(i, i)*m_shost(i, i);
                    }
                    snorm = std::sqrt(snorm);
                }
                else{snorm = 1.0;}

                if(trunc_mode == truncation_mode::singular_values_truncation)
                {
                    for(size_type i = 0; i < bond_dimension; ++i)
                    {
                        if(std::abs(m_shost(i, i)) > tol*snorm){++nb;}
                    }
                }
                else if(trunc_mode == truncation_mode::weight_truncation)
                {
                    real_type discarded_weight = 0;
                    nb = bond_dimension;
                    for(size_type i = 0; i < bond_dimension; ++i)
                    {
                        size_type ind = bond_dimension-(i+1);
                        discarded_weight += m_shost(ind, ind)*m_shost(ind, ind);
                        if(discarded_weight > tol*snorm)
                        {
                            nb = ind+1;
                            break;
                        }
                    }
                }
        
                nb = std::max(size_type(1), nb);
                bond_dimension = std::min(nb, bond_dimension);
            }
            else if(save_shost)
            {
                CALL_AND_HANDLE(m_shost = m_s, "Failed to copy singular values to host.");
            }

          
            if(A.shape(1) > 1 && A.shape(0) > 1)
            {
                if(bond_dimension < 2){bond_dimension = 2;}
            }

            if(m_s.size(0) == bond_dimension && m_s.size(1) == bond_dimension)  
            {
                //if A.shape(0) < A.shape(1) then the resultant U matrix is the square matrix.  For our implementation this matrix must be 
                //a rectangular matrix and so now we resize and zero pad the U and S matrices to make sure everything is correct.
                if(A.shape(0) < A.shape(1))
                {       
                    CALL_AND_HANDLE(pad_buffer(A, U, A.shape(1)), "Failed to pad buffer");
                }
                CALL_AND_HANDLE(V = m_s*m_temp, "Failed to assign rb matrix.");
            }
            ////now handle the case where we need to perform truncation
            ////first truncate s and temp so that the matrix multiplication can be performed as required
            else
            {
                //if A.shape(0) < A.shape(1) then the resultant U matrix is the square matrix.  For our implementation this matrix must be 
                //a rectangular matrix and so now we resize and zero pad the U and S matrices to make sure everything is correct.  If we are truncating then
                //in the case where we are truncating we only pad up to bond_dimension
                if(A.shape(0) < A.shape(1))
                {       
                    //if the shape is less than bond dimension then we pad up to bond-dimension in size - we don't do the full padding
                    CALL_AND_HANDLE(pad_buffer(A, U, A.shape(1)), "Failed to pad buffer");
                }
                CALL_AND_HANDLE(truncateU(U, bond_dimension), "Failed to truncate U matrix.");
                CALL_AND_HANDLE(truncateV(m_temp, bond_dimension), "Failed to truncate U matrix.");

                CALL_AND_HANDLE(V.resize(bond_dimension, V.size(1)), "Failed to resize V array.");
                m_s.resize(bond_dimension, bond_dimension);

                CALL_AND_HANDLE(V = m_s*m_temp, "Failed to compute V");
            }

            return bond_dimension;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying the decomposition engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the decomposition engine.");
        }
    }

protected:
    template <typename Atype, typename Utype>
    void pad_buffer(const Atype& A, Utype& U, size_type bond_dimension)
    {
        ASSERT(A.size() <= m_temp2.capacity(), "The second temporary matrix does not have sufficient capacity.");
        CALL_AND_HANDLE(m_temp2.resize(A.shape(0), bond_dimension), "Failed to resize the second temporary matrix so that it has the correct shape.");

        using memfill = linalg::memory::filler<real_type, backend>;
        try
        {
            backend::fill_matrix_block(U.buffer(), U.shape(0), bond_dimension, m_temp2.buffer(), m_temp2.shape(0), bond_dimension);
        }
        catch(const std::exception& ex)
        {
            RAISE_EXCEPTION("Failed to zero pad the U matrix so that it has the correct shape.");
        }

        CALL_AND_HANDLE(U.resize(U.shape(0), bond_dimension), "Failed to resize the U matrix.");

        //we might want to change this for a swap operation
        CALL_AND_HANDLE(U = m_temp2, "Failed to assign the U matrix.");
        
        
        CALL_AND_HANDLE(m_s.resize(bond_dimension, bond_dimension), "Failed to resize the S matrix.");

        try
        {
            memfill::fill(m_s.buffer()+A.shape(0), bond_dimension-A.shape(0), real_type(0.0));
        }
        catch(const std::exception& ex)
        {
            RAISE_EXCEPTION("Failed to zero pad the S matrix so that it has the correct shape.");
        }
    }

    template <typename Utype>
    void truncateU(Utype& U, size_type bond_dimension)
    {
        CALL_AND_HANDLE(m_temp2.resize(U.shape(0), bond_dimension), "Failed to resize temp buffer");
        try
        {
            backend::copy_matrix_subblock(U.shape(0), bond_dimension, U.buffer(), U.shape(1), m_temp2.buffer(), bond_dimension);
            
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to zero pad the U matrix so that it has the correct shape.");
        }
        CALL_AND_HANDLE(U.resize(U.shape(0), bond_dimension), "Failed to resize the U matrix.");
        CALL_AND_HANDLE(U = m_temp2, "Failed to assign the U matrix.");
    }

    template <typename Utype>
    void truncateV(Utype& V, size_type bond_dimension)
    {
        CALL_AND_HANDLE(m_temp2.resize(bond_dimension, V.shape(1)), "Failed to resize temp buffer");
        try
        {
            backend::copy_matrix_subblock( bond_dimension, V.shape(1), V.buffer(), V.shape(0), m_temp2.buffer(), V.shape(1));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to zero pad the U matrix so that it has the correct shape.");
        }
        CALL_AND_HANDLE(V.resize(bond_dimension, V.shape(1)), "Failed to resize the U matrix.");
        CALL_AND_HANDLE(V = m_temp2, "Failed to assign the U matrix.");
    }
public:


    template <typename Vtype>
    void transposeV(Vtype& V)
    {
        try
        {
            if(std::is_same<backend, linalg::blas_backend>::value && V.shape(0) == V.shape(1))
            {
                CALL_AND_HANDLE(V = trans(V), "Failed to evaluate inplace transpose of V matrix.");
            }
            else
            {
                std::cerr << "here" << std::endl;
                ASSERT(V.size() <= m_temp.capacity(), "The temporary matrix does not have sufficient capacity.");
                CALL_AND_HANDLE(m_temp = trans(V), "Failed to evaluate transpose of V matrix into temporary buffer.");
                CALL_AND_HANDLE(V = m_temp, "Failed to store transposed V matrix.");
            }
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying transposeV.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply transposeV on result of decomposition engine.");
        }
    }

    const dmat_type& S() const
    {
        return m_s;
    }

    const dmat_host_type& Shost() const
    {
        return m_shost;
    }

    svd_engine& svd(){return m_svd;}
protected:
    template <typename decomp_type, typename resize_obj>
    void resize_buffers(const resize_obj& A, bool use_capacity = false)
    {
        try
        {
            std::array<size_type, 2> max_dim{{0,0}};
            size_type temp2_cap = 0;
            //iterate over all nodes and determine the maximum dimension in each 
            for(const auto& a : A)
            {
                std::array<size_type, 2> dim = decomp_type::maximum_matrix_dimension_node(a(), use_capacity);
                for(size_type i = 0; i<2; ++i){if(dim[i] > max_dim[i]){max_dim[i] = dim[i];}}

                size_type node_size = use_capacity ? a().capacity() : a().size();
                if(node_size > temp2_cap){temp2_cap = node_size;}
            }

            //resize the svd object
            CALL_AND_HANDLE(m_svd.resize(max_dim[0], max_dim[1], false), "Failed to resize svd object.");

            //and the temporary buffer we use for constructing the temporary results into
            CALL_AND_HANDLE(m_temp.resize(max_dim[1], max_dim[1]), "Failed to resize temporary buffer.");
            CALL_AND_HANDLE(m_temp2.resize(1, temp2_cap), "Failed to resize second temporary buffer.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize decomposition engine object.");
        }
    }


    dmat_host_type m_shost;
    dmat_type m_s;
    matrix_type m_temp;
    matrix_type m_temp2;
    svd_engine m_svd;
};
}   //namespace orthogonality
}   //namespace ttns

#endif //TTNS_LIB_TTN_ORTHOGONALITY_DECOMPOSITION_ENGINE_HPP


