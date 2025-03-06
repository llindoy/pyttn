#ifndef LINALG_SPECIAL_FUNCTIONS_DETERMINANT_HPP
#define LINALG_SPECIAL_FUNCTIONS_DETERMINANT_HPP

#include "../decompositions/lu_decomposition/lu_decomposition.hpp"

namespace linalg
{
template <typename matrix_type>
class determinant<matrix_type, true, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, blas_backend>::value, void>::type>
{
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;

protected:
    matrix_type m_temp;
    vector<int, backend_type> m_ipiv;
    lu_decomposition<matrix_type> m_lu;

public:
    determinant(){}
    determinant(size_type n, bool use_temporary = true){CALL_AND_HANDLE(resize(n, use_temporary), "Failed to construct determinant engine.  Failed to resize temporary array.");}
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, 
                                 typename = typename std::enable_if<internal::valid_decomposition_matrix<matrix_type, value_type, backend_type>::value, void>::type >
    determinant(const mat_type& m, bool use_temporary = true)
    {   
        ASSERT(m.size(0) == m.size(1), "Failed to construct determinant engine.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.size(0), use_temporary), "Failed to construct determinant engine.  Failed to resize temporary array.");
    }

    void resize(size_t n, bool use_temporary = true)
    {
        if(use_temporary){CALL_AND_HANDLE(m_temp.resize(n, n), "Failed to resize determinant engine object.  Failed when resizing internal matrix.");}  
        CALL_AND_HANDLE(m_ipiv.resize(n), "Failed to resize determinant engine object.  Failed when resizing ipiv array.");
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type operator()(const mat_type& m)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute determinant.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute determinant.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute determinant.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute determinant.  LU decomposition failed.");
        return compute_determinant(m_temp);
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type operator()(mat_type& m, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute determinant.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute determinant.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute determinant.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute determinant.  LU decomposition failed.");

            return compute_determinant(m_temp);
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute determinant.  LU decomposition failed.");
            //now we compute the determinant from the LU decomposition
            return compute_determinant(m);
        }
    }

protected:
    template <typename mat_type>
    value_type compute_determinant(const mat_type& m)
    {
        value_type _determinant(1.0);
        for(size_type i=0; i<m_ipiv.size(); ++i)
        {
            _determinant *= m(i, i)*static_cast<typename mat_type::value_type>((static_cast<size_type>(m_ipiv(i)) != i+1) ? -1.0 : 1.0);
        }
        return _determinant;
    }
    
};  //class determinant
}   //namespace linalg


#ifdef PYTTN_BUILD_CUDA
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
namespace linalg
{
template <typename matrix_type>
class determinant<matrix_type, true, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, cuda_backend>::value, void>::type>
{
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;

protected:
    matrix_type m_temp;
    vector<value_type, backend_type> m_tred;
    vector<int, backend_type> m_ipiv;
    lu_decomposition<matrix_type> m_lu;

public:
    determinant(){}
    determinant(size_type n, bool use_temporary = true){CALL_AND_HANDLE(resize(n, use_temporary), "Failed to construct determinant engine.  Failed to resize temporary array.");}
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, 
                                 typename = typename std::enable_if<internal::valid_decomposition_matrix<matrix_type, value_type, backend_type>::value, void>::type >
    determinant(const mat_type& m, bool use_temporary = true)
    {   
        ASSERT(m.size(0) == m.size(1), "Failed to construct determinant engine.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.size(0), use_temporary), "Failed to construct determinant engine.  Failed to resize temporary array.");
    }

    void resize(size_t n, bool use_temporary = true)
    {
        if(use_temporary){CALL_AND_HANDLE(m_temp.resize(n, n), "Failed to resize determinant engine object.  Failed when resizing internal matrix.");}  
        CALL_AND_HANDLE(m_ipiv.resize(n), "Failed to resize determinant engine object.  Failed when resizing ipiv array.");
        CALL_AND_HANDLE(m_tred.resize(n), "Failed to resize determinant engine object.  Failed when resizing temporary reduction array.");
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type operator()(const mat_type& m)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute determinant.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute determinant.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute determinant.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute determinant.  LU decomposition failed.");
        return compute_determinant(m_temp);
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type operator()(mat_type& m, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute determinant.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute determinant.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute determinant.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute determinant.  LU decomposition failed.");

            return compute_determinant(m_temp);
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute determinant.  LU decomposition failed.");
            //now we compute the determinant from the LU decomposition
            return compute_determinant(m);
        }
    }

    template <typename mat_type>
    value_type compute_determinant(mat_type& m)
    {	
	size_type N = m.size(1);
        m_tred.fill(
            [] __host__ __device__ (size_type i, value_type* mbuf, int* ipiv, size_type n)
            {
                return mbuf[i*(n+1)]*((ipiv[i] != i+1) ? -1.0 : 1.0);
            },
            m.buffer(), m_ipiv.buffer(), N
        );
	thrust::device_ptr<value_type> temp = thrust::device_pointer_cast(m_tred.buffer());
        return thrust::reduce(temp, temp+N, value_type(1.0), thrust::multiplies<value_type>());
    }
    
};  //class determinant

}   //namespace linalg
#endif


#endif  //LINALG_SPECIAL_FUNCTIONS_DETERMINANT_HPP//

