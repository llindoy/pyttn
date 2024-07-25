#ifndef LINALG_DECOMPOSITIONS_EIGENSOLVER_SYMMETRIC_TRIDIAGONAL_MATRIX_HPP
#define LINALG_DECOMPOSITIONS_EIGENSOLVER_SYMMETRIC_TRIDIAGONAL_MATRIX_HPP

#include "eigensolver_base.hpp"

namespace linalg
{
namespace internal
{
template <typename matrix_type>
class symmetric_tridiagonal_eigensolver
{
    static_assert(is_symtridiag_matrix_type<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, blas_backend>::value && !is_complex<typename traits<matrix_type>::value_type>::value, "Invalid template parameter for symmetric_tridiagonal_eigensolver");
public:
    using int_type = blas_backend::blas_int_type;
    using size_type = blas_backend::size_type;
    using mem_trans = memory::transfer<blas_backend, blas_backend>;
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
protected:
    tensor<value_type, 1, blas_backend> m_E;
    tensor<value_type, 1, blas_backend> m_work;
public:
    symmetric_tridiagonal_eigensolver() {}
    symmetric_tridiagonal_eigensolver(size_type n, bool compute_eigenvectors = true, bool keep_input = true){CALL_AND_HANDLE(resize(n, compute_eigenvectors, keep_input), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
    symmetric_tridiagonal_eigensolver(const mat_type& mat, bool compute_eigenvectors = true, bool keep_input = true){CALL_AND_HANDLE(resize(mat.shape(0), compute_eigenvectors, keep_input), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}

    /////////////////////////////////////////////////////////////////////////////////////////
    //     Functions for resizing the internal buffers required for computing a symmetric  //
    //                      tridiagonal matrix eigendecomposition.                         //
    /////////////////////////////////////////////////////////////////////////////////////////
    void resize(size_type n, bool compute_eigenvectors = true, bool keep_input = true)
    {
        try
        {
            if(keep_input){if(n > m_E.size()+1){CALL_AND_HANDLE(m_E.resize(n-1), "Failed when resizing internal E buffer.");}}
            if(compute_eigenvectors){size_type worksize = 2*n-2;  if(worksize > m_work.size()){CALL_AND_HANDLE(m_work.resize(1 > worksize ? 1 : worksize), "Failed when resizing the internal working buffer.");}}
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize eigensolver object.");
        }
    }
    template <typename mat_type>
    valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type& mat, bool compute_eigenvectors = true, bool keep_input = true){CALL_AND_RETHROW(resize(mat.shape(0), compute_eigenvectors, keep_input));}


    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_E.clear(), "Failed when clearing the E array.");
            CALL_AND_HANDLE(m_work.clear(), "Failed when clearing the working array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear eigensolver object. ");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //   Functions for computing the eigendecomposition of a symmetric tridiagonal matrix  //
    /////////////////////////////////////////////////////////////////////////////////////////
    //functions for computing the eigenvalues of a symmetric tridiagonal matrix
    template <typename mat_type>
    typename std::enable_if<is_valid_decomp_matrix_type<matrix_type, mat_type>::value && traits<mat_type>::is_mutable, void>::type operator()(mat_type& mat)
    {
        try
        {
            char JOBZ = 'N';    int_type n = mat.shape(0);    value_type Z;    int_type ldz = 1;  value_type work;
            CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, mat.D(), mat.E(), &Z, ldz, &work), "Failed when making lapack call.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigenvalues of symmetric tridiagonal matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigenvalues of symmetric tridiagonal matrix. ");
        }
    }
    template <typename mat_type>
    typename std::enable_if<is_valid_decomp_matrix_type<matrix_type, mat_type>::value && traits<mat_type>::is_mutable, void>::type operator()(mat_type&& mat)
    {
        try
        {
            char JOBZ = 'N';    int_type n = mat.shape(0);    value_type Z;    int_type ldz = 1;  value_type work;
            CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, mat.D(), mat.E(), &Z, ldz, &work), "Failed when making lapack call.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigenvalues of symmetric tridiagonal matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigenvalues of symmetric tridiagonal matrix. ");
        }
    }


    //functions for computing the eigenvalues and right eigenvectors of a symmetric tridiagonal matrix
    template <typename mat_type, typename vals_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(const mat_type& mat, vals_type& eigs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, true), "Invalid inputs.");
            char JOBZ = 'N';    int_type n = mat.shape(0);    value_type Z;    int_type ldz = 1;  value_type work;

            CALL_AND_HANDLE(mem_trans::copy(mat.D(), mat.shape(0), eigs.buffer()) , "Failed to copy the diagonal component to a temporary buffer.");
            CALL_AND_HANDLE(mem_trans::copy(mat.E(), mat.shape(0)-1, m_E.buffer()), "Failed to copy the off-diagonal component to a temporary buffer.");
            CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, eigs.buffer(), m_E.buffer(), &Z, ldz, &work), "Failed when making lapack call.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigenvalues of symmetric tridiagonal matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigenvalues of symmetric tridiagonal matrix ");
        }
    }
    template <typename mat_type, typename vals_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(mat_type& mat, vals_type& eigs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, keep_inputs), "Invalid inputs.");
            char JOBZ = 'N';    int_type n = mat.shape(0);    value_type Z;    int_type ldz = 1;  value_type work;

            CALL_AND_HANDLE(mem_trans::copy(mat.D(), mat.shape(0), eigs.buffer()) , "Failed to copy the diagonal component to a temporary buffer.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(mem_trans::copy(mat.E(), mat.shape(0)-1, m_E.buffer()), "Failed to copy the off-diagonal component to a temporary buffer.");
                CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, eigs.buffer(), m_E.buffer(), &Z, ldz, &work), "Failed when making lapack call.");
            }
            else{CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, eigs.buffer(), mat.E(), &Z, ldz, &work), "Failed when making lapack call.");}
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigenvalues of symmetric tridiagonal matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigenvalues of symmetric tridiagonal matrix. ");
        }
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_type, vals_type, vecs_type> operator()(const mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, true), "Invalid inputs.");
            char JOBZ = 'V';    int_type n = mat.shape(0);    int_type ldz = vecs.shape(1);

            CALL_AND_HANDLE(mem_trans::copy(mat.D(), mat.shape(0), eigs.buffer()) , "Failed to copy the diagonal component to a temporary buffer.");
            CALL_AND_HANDLE(mem_trans::copy(mat.E(), mat.shape(0)-1, m_E.buffer()), "Failed to copy the off-diagonal component to a temporary buffer.");
            CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, eigs.buffer(), m_E.buffer(), vecs.buffer(), ldz, m_work.buffer()), "Failed when making lapack call.");
            CALL_AND_HANDLE(vecs = trans(vecs), "Failed when transposing the resulting vecs array to get the correct eigenvectors.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigendecomposition of symmetric tridiagonal matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigendecomposition of symmetric tridiagonal matrix. ");
        }
    }    
    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_type, vals_type, vecs_type> operator()(mat_type& mat, vals_type& eigs, vecs_type& vecs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, keep_inputs), "Invalid inputs.");
            CALL_AND_HANDLE(mem_trans::copy(mat.D(), mat.shape(0), eigs.buffer()) , "Failed to copy the diagonal component to a temporary buffer.");
            char JOBZ = 'V';    int_type n = mat.shape(0);    int_type ldz = vecs.shape(1);

            if(keep_inputs)
            {
                CALL_AND_HANDLE(mem_trans::copy(mat.E(), mat.shape(0)-1, m_E.buffer()), "FFailed to copy the off-diagonal component to a temporary buffer.");
                CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, eigs.buffer(), m_E.buffer(), vecs.buffer(), ldz, m_work.buffer()), "Failed when making lapack call.");
                CALL_AND_HANDLE(vecs = trans(vecs), "Failed when transposing the resulting vecs array to get the correct eigenvectors.");
            }
            else
            {
                CALL_AND_HANDLE(blas_backend::stev(JOBZ, n, eigs.buffer(), mat.E(), vecs.buffer(), ldz, m_work.buffer()), "Failed when making lapack call.");
                CALL_AND_HANDLE(vecs = trans(vecs), "Failed when transposing the resulting vecs array to get the correct eigenvectors.");
            }
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigendecomposition of symmetric tridiagonal matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigendecomposition of symmetric tridiagonal matrix. ");
        }
    }

protected:
    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
    //  engine for a symmetric tridiagonal matrix                                          //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type, typename vals_type>
    void validate(const mat_type& mat, vals_type& eigs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of symmetric tridiagonal matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(resize(mat.shape(0), false, use_matrix), "Failed to compute eigenvalues of symmetric tridiagonal matrix.  Failed to resize internal storage buffers.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    void validate(const mat_type& mat, vals_type& eigs, vecs_type& vecs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of symmetric tridiagonal matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs), "Failed to compute eigendecomposition of symmetric tridiagonal matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
        CALL_AND_HANDLE(resize(mat.shape(0), true, use_matrix), "Failed to compute eigenvalues of symmetric tridiagonal matrix.  Failed to resize internal storage buffers.");
    }

};
}   //namespace internal


//eigensolver object for symmetric tridiagonal matrix type that is not mutable
template <typename matrix_type>
class eigensolver
<
    matrix_type,        //matrix type
    typename std::enable_if
    <
        is_symtridiag_matrix_type<remove_cvref_t<matrix_type>>::value && 
        std::is_same<typename traits<remove_cvref_t<matrix_type>>::backend_type, blas_backend>::value && 
        !is_complex<typename traits<remove_cvref_t<matrix_type>>::value_type>::value,
    void>::type       //enabler
> : public internal::symmetric_tridiagonal_eigensolver<remove_cvref_t<matrix_type>>
{
public:
    using base_type = internal::symmetric_tridiagonal_eigensolver<remove_cvref_t<matrix_type>>;
    template <typename ... Args>  eigensolver(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};
}   //namespace linalg


#endif //LINALG_DECOMPOSITIONS_EIGENSOLVER_SYMMETRIC_TRIDIAGONAL_MATRIX_HPP//


