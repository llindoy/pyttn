#ifndef LINALG_DECOMPOSITIONS_SPARSE_ARNOLDI_ITERATION_HPP
#define LINALG_DECOMPOSITIONS_SPARSE_ARNOLDI_ITERATION_HPP

#include <limits>

#include "../../linalg_forward_decl.hpp"
#include "../../algebra/algebra.hpp"


namespace linalg
{

template <typename T, typename backend>
class arnoldi_iteration
{
public:
    using value_type = T;
    using real_type = typename get_real_type<T>::type;
    using backend_type = backend;
    using size_type = typename backend_type::size_type;
protected:
    size_type m_max_krylov_dim;
    size_type m_max_dim;
    size_type m_krylov_dim;
    size_type m_dim;
    size_type m_cur_krylov_dim;
    matrix<value_type, backend_type> m_Q;       //the krylov subspace vectors.  Here the rows of the matrix Q form the orthonormal basis
    vector<value_type, backend_type> m_w;
    matrix<value_type> m_H;                     //the upper hessenberg matrix representing the operator in the krylov subspace.  Here this is stored in column major order
                                                //this matrix is only the size of the active krylov subspace 
    matrix<value_type> m_Hv;                    //a buffer for storing the upper hessenberg matrix representation in the krylov subspac.  This buffer allows for the maximum 
                                                //krylov subspace dimension to be stored.

    real_type m_threshold;

    //two quantities useful for setting the error estimate
    real_type m_beta;

public:
    arnoldi_iteration() : m_max_krylov_dim(0), m_max_dim(0), m_krylov_dim(0), m_dim(0), m_threshold(1e-36), m_beta(0) {}
    arnoldi_iteration(size_type krylov_dim, size_type max_dim) 
        try : m_max_krylov_dim(0), m_max_dim(0), m_threshold(1e-36), m_beta(0)
    {
        resize(krylov_dim, max_dim);
    }
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct arnoldi iteration object.");}

    arnoldi_iteration(const arnoldi_iteration& o) = default;
    arnoldi_iteration(arnoldi_iteration&& o) = default;

    arnoldi_iteration& operator=(const arnoldi_iteration& o) = default;
    arnoldi_iteration& operator=(arnoldi_iteration&& o) = default;
    
    void resize(size_type krylov_dim, size_type dim)
    {
        try
        {
            if(krylov_dim > m_max_krylov_dim){m_max_krylov_dim = krylov_dim;}
            if(dim > m_max_dim){m_max_dim = dim;}

            m_krylov_dim = krylov_dim;
            m_dim = dim;

            CALL_AND_HANDLE(m_Q.resize(krylov_dim+1, dim), "Failed to resize the arnoldi vectors.");
            CALL_AND_HANDLE(m_w.resize(dim), "Failed to resize the temporary arnoldi vector.");
            CALL_AND_HANDLE(m_H.resize(krylov_dim, krylov_dim), "Failed to resize the resultant upper hessenberg matrix storing the operator in the krylov subspace.");
            CALL_AND_HANDLE(m_Hv.resize(krylov_dim, krylov_dim+1), "Failed to resize the resultant upper hessenberg matrix storing the operator in the krylov subspace.");
        }        
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize arnoldi iteration object.");
        }
    }

    void reallocate(size_type krylov_dim, size_type dim)
    {
        try
        {
            m_max_krylov_dim = krylov_dim;
            m_max_dim = dim;

            m_krylov_dim = krylov_dim;
            m_dim = dim;

            CALL_AND_HANDLE(m_Q.reallocate(krylov_dim+1, dim), "Failed to reallocate the arnoldi vectors.");
            CALL_AND_HANDLE(m_w.reallocate(dim), "Failed to reallocate the temporary arnoldi vector.");
            CALL_AND_HANDLE(m_H.reallocate(krylov_dim, krylov_dim), "Failed to reallocate the resultant upper hessenberg matrix storing the operator in the krylov subspace.");
            CALL_AND_HANDLE(m_Hv.reallocate(krylov_dim, krylov_dim+1), "Failed to reallocate the resultant upper hessenberg matrix storing the operator in the krylov subspace.");
        }        
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to reallocate arnoldi iteration object.");
        }
    }

    void deallocate()
    {
        try
        {
            m_krylov_dim = 0;   m_dim = 0;    m_max_krylov_dim = 0;   m_max_dim = 0;
            CALL_AND_HANDLE(m_Q.deallocate(), "Failed to deallocate the arnoldi vectors.");
            CALL_AND_HANDLE(m_w.deallocate(), "Failed to deallocate the temporary arnoldi vector.");
            CALL_AND_HANDLE(m_H.deallocate(), "Failed to deallocate the resultant upper hessenberg matrix storing the operator in the krylov subspace.");
            CALL_AND_HANDLE(m_Hv.deallocate(), "Failed to deallocate the resultant upper hessenberg matrix storing the operator in the krylov subspace.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to deallocate arnoldi iteration object.");
        }
    }

    void clear()
    {
        CALL_AND_RETHROW(deallocate());
    }

    void reset_zeros()
    {
        m_Q.fill_zeros();
        m_w.fill_zeros();
        m_H.fill_zeros();
        m_Hv.fill_zeros();
        m_beta = 0.0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Function for computing the krylov subspace using the inbuilt scalar multiplication function for a    //
    // matrix and a vector.                                                                                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename vec_type, typename ... Args>
    bool operator()(const vec_type& vec, real_type scale, Args&& ... args) 
    {
        CALL_AND_HANDLE(return partial_krylov_step(vec, scale, 0, m_krylov_dim, std::forward<Args>(args)...), "Failed to compute the krylov subspace.");
    }

    template <typename vec_type, typename ... Args>
    bool partial_krylov_step(const vec_type& vec, real_type scale, size_type istart, size_type iend, Args&& ... args) 
    {
        try
        {
            ASSERT(m_krylov_dim != 0, "Failed to compute the partial krylov subspace. Cannot construct a krylov subspace with dim of 0.");
            ASSERT(iend > 0, "Krylov subspace with one vector is useless.");
            ASSERT(istart <= iend, "Failed to compute the partial krylov subspace.  The starting index must be smaller than the final index.");
            ASSERT(iend <= m_krylov_dim, "Failed to compute the partial krylov subspace.  The final index is larger than the maximum krylov subspace dimension.");

            
            CALL_AND_HANDLE(reset_krylov_subspace_dimension(vec.size()), "Failed to compute the partial krylov subspace.  Failed to initialise the krylov subspace vector storage.");
            bool iteration_completed_early = false;

            using std::sqrt;

            auto wp = m_w.reinterpret_shape(vec.shape());
            size_t cur_krylov_dim = 0;
            for(size_type i=istart; i <= iend && !iteration_completed_early; ++i)
            {
                cur_krylov_dim=i;
                if(i == 0)
                {
                    auto q0 = m_Q[0];  q0.set_buffer(vec);  real_type fnorm = sqrt(real(dot_product(conj(q0), q0)));  
                    m_beta = fnorm;
                    q0 = q0/fnorm;
                }
                else
                {
                    auto qp = m_Q[i-1];      //get the previous vector
                    auto qi = m_Q[i];        //and the new vector we are trying to construct

                    auto rqp = qp.reinterpret_shape(vec.shape());       //get this object shaped correctly for apply the operator defined by args

                    //now apply the operator defined by args that is potentially scaled
                    CALL_AND_HANDLE(compute_action(rqp, wp, std::forward<Args>(args)...), "Failed to compute the partial krylov subspace.  Failed to evaluate action on vector.");
                    m_w *= scale;       // and handle the scaling

                    //now we orthogonalise this vector against the current krylov subspace
                    for(size_type j=0; j<i; ++j)
                    {
                        auto qj = m_Q[j];
                        CALL_AND_HANDLE(m_Hv(i-1, j) = dot_product(conj(qj), m_w), "Failed to compute the partial krylov subspace.  Failed to compute overlap necessary for modified Gram-Schmitd.");
                        CALL_AND_HANDLE(m_w -= m_Hv(i-1, j)*qj, "Failed to compute partial krylov subspace.  Failed to orthogonalise vector.");
                    }
                    real_type norm = sqrt(real(dot_product(conj(m_w), m_w)));
                    cur_krylov_dim = i; 
                    m_Hv(i-1, i) = norm;
                    if(norm < m_threshold){iteration_completed_early = true;}
                    else
                    {
                        qi = m_w/norm;
                    }
                }
            }
            CALL_AND_HANDLE(finalise_krylov_rep(cur_krylov_dim), "Failed to compute partial krylov subspace.  Failed when finalising the iteration.");
            return iteration_completed_early;
        }        
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing the partial krylov subspace.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute the partial krylov subspace.");
        }
    }
    
    const real_type& beta() const{return m_beta;}
    real_type hk1k() const{return std::real(m_Hv(m_cur_krylov_dim-1, m_cur_krylov_dim));}
    real_type& threshold(){return m_threshold;}
    const real_type& threshold() const{return m_threshold;}
    const size_type& current_krylov_dim() const{return m_cur_krylov_dim;}

    const matrix<value_type, backend_type>& Q()const {return m_Q;}
    matrix<value_type, backend_type>& Q(){return m_Q;}
    upper_hessenberg_matrix<const T, blas_backend> H()const {return upper_hessenberg_view(m_H, MATRIX_ORDERING::COLUMN_MAJOR);}
    upper_hessenberg_matrix<T, blas_backend> H(){return upper_hessenberg_view(m_H, MATRIX_ORDERING::COLUMN_MAJOR);}

protected:
    void reset_krylov_subspace_dimension(size_type maxsize)
    {
        if(maxsize > m_max_dim){}//print an info message telling the user that the krylov subspace will be reshaped resized"
        //resize the m_Q vector 
        CALL_AND_HANDLE(m_Q.resize(m_krylov_dim, maxsize), "Failed to compute the krylov subspace.  Failed to resize the arnoldi vectors.");
    }

public:
    void finalise_krylov_rep(size_type cur_krylov_dim)
    {
        using std::real;
        //now we resize the upper hessenberg matrix and the arnoldi vectors to be the actual size used
        CALL_AND_HANDLE(m_Q.resize(cur_krylov_dim, m_Q.shape(1)), "Failed to compute the krylov subspace representation of mat acting on vec.  Failed to resize the arnoldi vectors matrix so that it is the correct shape.");
        CALL_AND_HANDLE(m_H.resize(cur_krylov_dim, cur_krylov_dim), "Failed to compute the krylov subspace representation of mat acting on vec.  Failed to resize the result matrix so that it is the correct shape.");
        m_H.fill_zeros();
        //and copy the working array into the upper hessenberg matrix in column major order
        for(size_type i=0; i < cur_krylov_dim; ++i)
        {
            for(size_type j=0; j<=i; ++j){m_H(i, j) = m_Hv(i, j);}
            if(i+1 != cur_krylov_dim){m_H(i, i+1) = m_Hv(i, i+1);}
        }
        m_cur_krylov_dim = cur_krylov_dim;
    }

public:
    //action of a matrix on a vector 
    template <typename vec_type, typename mat_type, typename res_type>
    typename std::enable_if<is_dense_tensor<vec_type>::value && is_linalg_object<mat_type>::value, void>::type compute_action(const vec_type& vec, res_type& r, const mat_type& mat) 
    {
        CALL_AND_RETHROW(r = mat*vec);
    }

    template <typename vtype, typename vec_type, typename mat_type, typename res_type>
    typename std::enable_if<is_number<vtype>::value && is_dense_tensor<vec_type>::value && is_linalg_object<mat_type>::value, void>::type compute_action(const vec_type& vec, res_type& r, const vtype& v, const mat_type& mat) 
    {
        CALL_AND_RETHROW(r = v*mat*vec);
    }

    //action of a function on a vector
    template <typename vec_type, typename res_type, typename Func, typename ... Args>
    typename std::enable_if<is_dense_tensor<vec_type>::value && !is_linalg_object<Func>::value, void>::type compute_action(const vec_type& vec, res_type& r, Func&& f, Args&&... args)
    {
        CALL_AND_RETHROW(f(vec, std::forward<Args>(args)..., r))
    }
};

}   //namespace linalg

#endif  //LINALG_DECOMPOSITIONS_SPARSE_ARNOLDI_ITERATION_HPP//

