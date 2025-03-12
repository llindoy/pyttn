#ifndef LINALG_DECOMPOSITIONS_TRIDIAGONALISATION_HERMITIAN_HPP
#define LINALG_DECOMPOSITIONS_TRIDIAGONALISATION_HERMITIAN_HPP

#include "tridiagonalisation_base.hpp"

namespace linalg
{
namespace internal
{

//implementation of the general generalised_eigensolver for real valued matrices
template <typename matrix_type>
class tridiagonalisation_hermitian
{
    static_assert(is_dense_matrix<matrix_type>::value, "Invalid template parameter for tridiagonalisation_hermitian");
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using real_type = typename get_real_type<value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;
    using int_type = typename backend_type::int_type;

protected:
    tensor<value_type, 1, backend_type> m_work;
    tensor<value_type, 1, backend_type> m_tau;
    tensor<value_type, 2, backend_type> m_mat;

public:
    tridiagonalisation_hermitian() {}
    tridiagonalisation_hermitian(size_type n, bool requires_matrix = true){CALL_AND_HANDLE(resize(n, requires_matrix), "Failed construct generalised_eigensolver with preallocated size.  Failed when resizing buffers.");}
    template <typename A_type, typename = typename std::enable_if<is_tensor<A_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, A_type, void>>
    tridiagonalisation_hermitian(const A_type& A, bool requires_matrix = true){CALL_AND_HANDLE(resize(A.shape(0), requires_matrix), "Failed construct generalised_eigensolver with preallocated size.  Failed when resizing buffers.");}

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for resizing the internal buffers required for computing a general       //
    //  generalised eigendecomposition.                                                    //
    /////////////////////////////////////////////////////////////////////////////////////////
    void resize(size_type n, bool requires_matrix = true)
    {
        try
        {
            if(requires_matrix)
            {
                CALL_AND_HANDLE(m_mat.resize(n, n), "Failed when resizing internal matrix.");
            }  
            CALL_AND_HANDLE(m_tau.resize(n-1), "Failed when resizing internal matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize generalised_eigensolver object.");
        }
    }
    template <typename A_type>
    valid_decomp_matrix_type<matrix_type, A_type, void> resize(const A_type& A, bool requires_matrix = true){CALL_AND_RETHROW(resize(A.shape(0), requires_matrix));}
    void resize_workspace(size_type worksize){if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of generalised_eigensolver object.");}}


    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_mat.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_tau.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_work.clear(), "Failed when clearing the working array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear generalised_eigensolver object.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for computing the eigendecomposition of a general matrix                 //
    /////////////////////////////////////////////////////////////////////////////////////////
    //functions for computing the eigenvalues of a general matrix
    template <typename mat_type>
    valid_decomp_matrix_type<matrix_type, mat_type, void> operator()(const mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, tri, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
            CALL_AND_RETHROW(compute(m_mat, tri));
        }
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating tridiagonalisation.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigenvalues.");
        }
    }

    template <typename mat_type>
    valid_decomp_matrix_type<matrix_type, mat_type, void> operator()(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, tri, true), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
                CALL_AND_RETHROW(compute(m_mat, tri));
            }
            else{CALL_AND_RETHROW(compute(mat, tri));}
        }
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigendecomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }

    //functions for computing the eigenvalues and right eigenvectors of a general matrix
    template <typename mat_type, typename Qtype>
    valid_decomp_func<matrix_type, mat_type, void, validate_vecs_type, Qtype> operator()(const mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri, Qtype& Q)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, tri, Q, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_mat, tri, Q));
        }
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigendecomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }

    template <typename mat_type, typename Qtype>
    valid_decomp_func<matrix_type, mat_type, void, validate_vecs_type, Qtype> operator()(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri, Qtype& Q, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, tri, Q, true), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_mat, tri, Q));
            }
            else
            {
                CALL_AND_RETHROW(compute(mat, tri, Q));
            }
        }
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigendecomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for determining the optimal workspace for eigendecomposition             //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type>
    valid_decomp_matrix_type<matrix_type, mat_type, size_type> query_work_space(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri)
    {
        try
        {
            char UPLO = 'U';    int_type N = mat.shape(0);   int_type LDA = mat.shape(1); 
            value_type worksize;    int_type lwork = -1;
            CALL_AND_HANDLE(blas_backend::sytrd(UPLO, N, mat.buffer(), LDA, tri.D(), tri.E(), m_tau.buffer(), &worksize, lwork), "Failed to query the optimal workspace for the eigensolver.");
            CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename mat_type, typename Qtype>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_vecs_type, Qtype> query_work_space(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri, Qtype& Q)
    {
        try
        {
            char UPLO = 'U';    int_type N = mat.shape(0);   int_type LDA = mat.shape(1); 
            value_type worksize;    int_type lwork = -1;
            CALL_AND_HANDLE(blas_backend::sytrd(UPLO, N, mat.buffer(), LDA, tri.D(), tri.E(), m_tau.buffer(), &worksize, lwork), "Failed to query the optimal workspace.");

            char SIDE = 'L'; char TRANS = 'T';  
            value_type worksize2;    lwork = -1;
            CALL_AND_HANDLE(blas_backend::ormtr(SIDE, UPLO, TRANS, N, N, mat.buffer(), LDA, m_tau.buffer(), Q.buffer(), N, &worksize2, lwork), "Failed to query the optimal workspac.");

            if(worksize2 > worksize){CALL_AND_RETHROW(return internal::worksize_as_integer(worksize2));}
            else{CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));}
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for computing the eigendecomposition of a general matrix          //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type>
    void compute(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, tri), "Failed to compute tridiagonal reduction of hermitian matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute tridiagonal reduction of hermitian matrix. Failed to resize the optimal workspace array.");
        
        char UPLO = 'U';    int_type N = mat.shape(0);   int_type LDA = mat.shape(1); int_type lwork = m_work.size();
        CALL_AND_HANDLE(blas_backend::sytrd(UPLO, N, mat.buffer(), LDA, tri.D(), tri.E(), m_tau.buffer(), m_work.buffer(), lwork), "Failed to query the optimal workspace for the eigensolver.");
    }

    template <typename mat_type, typename Qtype>
    void compute(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri, Qtype& Q)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, tri, Q), "Failed to compute tridiagonal reduction of hermitian matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute tridiagonal reduction of hermitian matrix. Failed to resize the optimal workspace array.");
        
        char UPLO = 'U';    int_type N = mat.shape(0);   int_type LDA = mat.shape(1); int_type lwork = m_work.size();
        CALL_AND_HANDLE(blas_backend::sytrd(UPLO, N, mat.buffer(), LDA, tri.D(), tri.E(), m_tau.buffer(), m_work.buffer(), lwork), "Failed to query the optimal workspace for the eigensolver.");

        Q.fill_zeros();
        for(size_t i = 0; i < N; ++i){Q(i, i) = 1.0;}

        char SIDE = 'L'; char TRANS = 'T';  
        CALL_AND_HANDLE(blas_backend::ormtr(SIDE, UPLO, TRANS, N, N, mat.buffer(), LDA, m_tau.buffer(), Q.buffer(), N, m_work.buffer(), lwork), "Failed to query the optimal workspac.");
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
    //  engine for a general matrix                                                        //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type>
    void validate(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri, bool use_matrix)
    {
        ASSERT(mat.size(0) == mat.size(1), "Failed to compute tridiagonal reduction of hermitian matrix. The matrix was not square.");
        
        CALL_AND_HANDLE(internal::tridiagonalisation_result_validation::tridiag(mat, tri), "Failed to compute tridiagonal reduction of hermitian matrix.  Failed to ensure the correct shape of the tridiagonal matrix.");

        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute tridiagonal reduction of hermitian matrix.  Failed resize buffers.");
    }

    template <typename mat_type, typename Qtype>
    void validate(mat_type& mat, symmetric_tridiagonal_matrix<real_type, backend_type>& tri, Qtype& Q)
    {
        ASSERT(mat.size(0) == mat.size(1), "Failed to compute tridiagonal reduction of hermitian matrix.  The matrix was not square.");
        CALL_AND_HANDLE(internal::tridiagonalisation_result_validation::tridiag(mat, tri), "Failed to compute tridiagonal reduction of hermitian matrix.  Failed to ensure the correct shape of the tridiagonal matrix.");
        CALL_AND_HANDLE(internal::tridiagonalisation_result_validation::sim_trans(mat, Q), "Failed to compute tridiagonal reduction of hermitian matrixx.  Failed to ensure the correct shape of the similarity transformation matrix.");
        CALL_AND_HANDLE(resize(mat.shape(0), true), "Failed to compute tridiagonal reduction of hermitian matrix.  Failed resize buffers.");
    }
};

}   //namespace internal

//Tridiagonal matrix solver
template <typename matrix_type>
class tridiagonalisation
<matrix_type, 
    typename std::enable_if
    <
        is_dense_matrix<remove_cvref_t<matrix_type>>::value && is_hermitian_matrix<remove_cvref_t<matrix_type>>::value &&
        std::is_same<typename traits<remove_cvref_t<matrix_type>>::backend_type, blas_backend>::value,
        void
    >::type
> : public internal::tridiagonalisation_hermitian<remove_cvref_t<matrix_type>>
{
public:
    using base_type = internal::tridiagonalisation_hermitian<remove_cvref_t<matrix_type>>;
    using backend_type = typename traits<remove_cvref_t<matrix_type>>::backend_type;
    using size_type = typename base_type::size_type;
    using value_type = typename base_type::value_type;
    template <typename ... Args>  tridiagonalisation(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};
}   //namespace linalg


#endif //LINALG_DECOMPOSITIONS_TRIDIAGONALISATION_HERMITIAN_HPP



