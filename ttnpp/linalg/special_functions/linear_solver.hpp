#ifndef LINALG_SPECIAL_FUNCTIONS_LINEAR_SOLVER_HPP
#define LINALG_SPECIAL_FUNCTIONS_LINEAR_SOLVER_HPP

#include "../decompositions/lu_decomposition/lu_decomposition.hpp"

namespace linalg
{
template <typename matrix_type>
class linear_solver<matrix_type, true, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, blas_backend>::value, void>::type>
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
    linear_solver(){}
    linear_solver(size_type n, bool use_temporary = true){CALL_AND_HANDLE(resize(n, use_temporary), "Failed to construct linear_solver engine.  Failed to resize temporary array.");}
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, 
                                 typename = typename std::enable_if<internal::valid_decomposition_matrix<matrix_type, value_type, backend_type>::value, void>::type >
    linear_solver(const mat_type& m, bool use_temporary = true)
    {   
        ASSERT(m.size(0) == m.size(1), "Failed to construct linear_solver engine.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.size(0), use_temporary), "Failed to construct linear_solver engine.  Failed to resize temporary array.");
    }

    void resize(size_t n, bool use_temporary = true)
    {
        if(use_temporary){CALL_AND_HANDLE(m_temp.resize(n, n), "Failed to resize linear_solver engine object.  Failed when resizing internal matrix.");}  
        CALL_AND_HANDLE(m_ipiv.resize(n), "Failed to resize linear_solver engine object.  Failed when resizing ipiv array.");
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type determinant(const mat_type& m)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
        return compute_determinant(m_temp);
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type determinant(mat_type& m, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");

            return compute_determinant(m_temp);
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            //now we compute the linear_solver from the LU decomposition
            return compute_determinant(m);
        }
    }

    template <typename mat_type, typename vec_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type apply_lu(const mat_type& m, vec_type& B)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");

        CALL_AND_HANDLE(blas_backend::getrs('T', m.size(1), 1, m.buffer(), m.size(1), m_ipiv.buffer(), B.buffer(), B.size()), "Failed to solve linear system.  Lapack call failed.");
    }

    template <typename mat_type, typename vec_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(const mat_type& m, vec_type& B)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");

        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
        CALL_AND_HANDLE(blas_backend::getrs('T', m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), B.buffer(), B.size()), "Failed to solve linear system.  Lapack call failed.");
    }

    template <typename mat_type, typename vec_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(mat_type& m, vec_type& B, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            CALL_AND_HANDLE(blas_backend::getrs('T', m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), B.buffer(), B.size()), "Failed to solve linear system.  Lapack call failed.");
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            //now we compute the linear_solver from the LU decomposition
            CALL_AND_HANDLE(blas_backend::getrs('T', m.size(1), 1, m.buffer(), m.size(1), m_ipiv.buffer(), B.buffer(), B.size()), "Failed to solve linear system.  Lapack call failed.");
        }
    }
    template <typename mat_type, typename vec_type, typename x_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(const mat_type& m, x_type& x, const vec_type& B)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        ASSERT(x.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
        x = B;
        CALL_AND_HANDLE(blas_backend::getrs('T', m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), x.buffer(), B.size()), "Failed to solve linear system.  Lapack call failed.");
    }


    template <typename mat_type, typename vec_type, typename x_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(mat_type& m, x_type& x, const vec_type& B, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        ASSERT(x.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            x = B;
            CALL_AND_HANDLE(blas_backend::getrs('T', m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), x.buffer(), B.size()), "Failed to solve linear system.  Lapack call failed.");
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            //now we compute the linear_solver from the LU decomposition
            x = B;
            CALL_AND_HANDLE(blas_backend::getrs('T', m.size(1), 1, m.buffer(), m.size(1), m_ipiv.buffer(), x.buffer(), B.size()), "Failed to solve linear system.  Lapack call failed.");
        }
    }

protected:
    template <typename mat_type>
    value_type compute_determinant(const mat_type& m)
    {
        value_type _linear_solver(1.0);
        for(size_type i=0; i<m_ipiv.size(); ++i)
        {
            _linear_solver *= m(i, i)*static_cast<typename mat_type::value_type>((static_cast<size_type>(m_ipiv(i)) != i+1) ? -1.0 : 1.0);
        }
        return _linear_solver;
    }
    
};  //class linear_solver
}   //namespace linalg


#ifdef PYTTN_BUILD_CUDA
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
namespace linalg
{
template <typename matrix_type>
class linear_solver<matrix_type, true, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, cuda_backend>::value, void>::type>
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

    tensor<int, 1, cuda_backend> m_gpu_info;
    tensor<int, 1> m_cpu_info;
public:
    linear_solver(){}
    linear_solver(size_type n, bool use_temporary = true){CALL_AND_HANDLE(resize(n, use_temporary), "Failed to construct linear_solver engine.  Failed to resize temporary array.");}
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, 
                                 typename = typename std::enable_if<internal::valid_decomposition_matrix<matrix_type, value_type, backend_type>::value, void>::type >
    linear_solver(const mat_type& m, bool use_temporary = true)
    {   
        ASSERT(m.size(0) == m.size(1), "Failed to construct linear_solver engine.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.size(0), use_temporary), "Failed to construct linear_solver engine.  Failed to resize temporary array.");
    }

    void resize(size_t n, bool use_temporary = true)
    {
        if(use_temporary){CALL_AND_HANDLE(m_temp.resize(n, n), "Failed to resize linear_solver engine object.  Failed when resizing internal matrix.");}  
        CALL_AND_HANDLE(m_ipiv.resize(n), "Failed to resize linear_solver engine object.  Failed when resizing ipiv array.");
        CALL_AND_HANDLE(m_tred.resize(n), "Failed to resize linear_solver engine object.  Failed when resizing temporary reduction array.");
        m_gpu_info.resize(1);
        m_cpu_info.resize(1);
    }

    template <typename mat_type, typename vec_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type apply_lu(mat_type& m, vec_type& B)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(cuda_backend::getrf(cuda_backend::op_t, m.size(1), 1, m.buffer(), m.size(1), m_ipiv.buffer(), B.buffer(), B.size(), m_gpu_info.buffer()), "Failed to solve linear system.  Lapack call failed.");
        m_cpu_info = m_gpu_info;
        ASSERT(m_cpu_info[0] == 0, "Invalid return code from getrs.");
    }

    template <typename mat_type, typename vec_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(const mat_type& m, vec_type& B)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
        CALL_AND_HANDLE(cuda_backend::getrf(cuda_backend::op_t, m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), B.buffer(), B.size(), m_gpu_info.buffer()), "Failed to solve linear system.  Lapack call failed.");
    }

    template <typename mat_type, typename vec_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(mat_type& m, vec_type& B, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");

            CALL_AND_HANDLE(cuda_backend::getrf(cuda_backend::op_t, m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), B.buffer(), B.size(), m_gpu_info.buffer()), "Failed to solve linear system.  Lapack call failed.");
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            //now we compute the linear_solver from the LU decomposition
            CALL_AND_HANDLE(cuda_backend::getrf(cuda_backend::op_t, m.size(1), 1, m.buffer(), m.size(1), m_ipiv.buffer(), B.buffer(), B.size(), m_gpu_info.buffer()), "Failed to solve linear system.  Lapack call failed.");
        }
    }

    template <typename mat_type, typename vec_type, typename x_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(const mat_type& m, x_type& X, const vec_type& B)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        ASSERT(X.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
        X = B;
        CALL_AND_HANDLE(cuda_backend::getrf(cuda_backend::op_t, m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), X.buffer(), B.size(), m_gpu_info.buffer()), "Failed to solve linear system.  Lapack call failed.");
    }


    template <typename mat_type, typename vec_type, typename x_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type operator()(mat_type& m, x_type& X, const vec_type& B, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        ASSERT(B.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        ASSERT(X.shape(0) == m.shape(0), "Failed to compute linear_solver.  The input vector is not compatible with the input matrix.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            X = B;
            CALL_AND_HANDLE(cuda_backend::getrf(cuda_backend::op_t, m_temp.size(1), 1, m_temp.buffer(), m_temp.size(1), m_ipiv.buffer(), X.buffer(), B.size(), m_gpu_info.buffer()), "Failed to solve linear system.  Lapack call failed.");
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            //now we compute the linear_solver from the LU decomposition
            X = B;
            CALL_AND_HANDLE(cuda_backend::getrf(cuda_backend::op_t, m.size(1), 1, m.buffer(), m.size(1), m_ipiv.buffer(), X.buffer(), B.size(), m_gpu_info.buffer()), "Failed to solve linear system.  Lapack call failed.");
        }
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type determinant(const mat_type& m)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), true), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
        CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
        return compute_determinant(m_temp);
    }

    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, value_type>::type determinant(mat_type& m, bool keep_input = true)
    {
        ASSERT(m.shape(0) == m.shape(1), "Failed to compute linear_solver.  The input matrix is not square.");
        CALL_AND_HANDLE(resize(m.shape(0), keep_input), "Failed to compute linear_solver.  Failed to resize the temporary buffers.");
        if(keep_input)
        {
            CALL_AND_HANDLE(m_temp = m, "Failed to compute linear_solver.  Failed to copy array into temporary array.");
            CALL_AND_HANDLE(m_lu(m, m_temp, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");

            return compute_determinant(m_temp);
        }
        else
        {
            CALL_AND_HANDLE(m_lu(m, m_ipiv), "Failed to compute linear_solver.  LU decomposition failed.");
            //now we compute the linear_solver from the LU decomposition
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
    
};  //class linear_solver

}   //namespace linalg
#endif


#endif  //LINALG_SPECIAL_FUNCTIONS_LINEAR_SOLVER_HPP//

