#ifndef LINALG_DECOMPOSITIONS_EIGENSOLVER_HERMITIAN_MATRIX_HPP
#define LINALG_DECOMPOSITIONS_EIGENSOLVER_HERMITIAN_MATRIX_HPP

#include "eigensolver_base.hpp"

namespace linalg
{
namespace internal
{

template <typename T, typename backend> struct hermitian_eigensolver_helper;

template <typename T> 
struct hermitian_eigensolver_helper<T, blas_backend>
{
    using int_type = blas_backend::int_type;
    static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to initialise hermitian eigensolver working space object.");
    using size_type = blas_backend::size_type;

    struct additional_working
    {
        void resize(size_type /* n */){}
        void clear(){}
    };

    static inline void call(const char JOBZ, const char UPLO, const int_type N, T* A, const int_type LDA, T* W, T* WORK, const int_type LWORK, additional_working& /* working */)
    {
        CALL_AND_RETHROW(blas_backend::heev(JOBZ, UPLO, N, A, LDA, W, WORK, LWORK););
    }
    static inline int_type query_worksize(const char JOBZ, const char UPLO, const int_type N, T* A, const int_type LDA, T* W, additional_working& /* working */)
    {
        T worksize;    int_type lwork = -1;
        CALL_AND_HANDLE(blas_backend::heev(JOBZ, UPLO, N, A, LDA, W, &worksize, lwork), "Failed to query the optimal workspace for the eigensolver.");
        CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
    }
};


template <typename T> 
struct hermitian_eigensolver_helper<complex<T>, blas_backend>
{
    using int_type = blas_backend::int_type;
    static_assert(is_number<T>::value, "Failed to initialise hermitian eigensolver working space object.");
    using size_type = blas_backend::size_type;

    struct additional_working
    {
        tensor<T, 1> m_rwork;
        void resize(size_type n){CALL_AND_RETHROW(m_rwork.resize(n > 0 ? 3*n-2 : 1));}
        void clear(){CALL_AND_RETHROW(m_rwork.clear());}
    };

    static inline void call(const char JOBZ, const char UPLO, const int_type N, complex<T>* A, const int_type LDA, T* W, complex<T>* WORK, const int_type LWORK, additional_working& working)
    {
        CALL_AND_RETHROW(blas_backend::heev(JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, working.m_rwork.buffer()););
    }
    static inline int_type query_worksize(const char JOBZ, const char UPLO, const int_type N, complex<T>* A, const int_type LDA, T* W, additional_working& working)
    {
        complex<T> worksize;    int_type lwork = -1;
        CALL_AND_HANDLE(blas_backend::heev(JOBZ, UPLO, N, A, LDA, W, &worksize, lwork, working.m_rwork.buffer()), "Failed to query the optimal workspace for the eigensolver.");
        CALL_AND_RETHROW(return internal::worksize_as_integer(worksize));
    }

};

#ifdef __NVCC__
template <typename T>
struct hermitian_eigensolver_helper<T, cuda_backend>
{
    static_assert(is_number<T>::value , "Failed to initialise hermitian eigensolver working space object.");
    using size_type = cuda_backend::size_type;
    using int_type = cuda_backend::int_type;

    struct additional_working
    {
        tensor<int, 1, cuda_backend> m_gpu_info;
        tensor<int, 1> m_cpu_info;
        void resize(size_type n){if(m_gpu_info.size() == 0){m_gpu_info.resize(1);   m_cpu_info.resize(1);}}
        void clear(){}
    };

    static inline void call(const char JOBZ, const char UPLO, const int_type N, T* A, const int_type LDA, typename get_real_type<T>::type* W, T* WORK, const int_type LWORK, additional_working& working)
    {
        int_type n = N; int_type lda = LDA; int_type lwork = LWORK;
        cusolverEigMode_t jobz; CALL_AND_RETHROW(jobz = get_jobz(JOBZ));
        cublasFillMode_t uplo; CALL_AND_RETHROW(uplo = get_uplo(UPLO));

        CALL_AND_RETHROW(cuda_backend::heev(jobz, uplo, n, A, lda, W, WORK, lwork, working.m_gpu_info.buffer());)
        working.m_cpu_info = working.m_gpu_info;    CALL_AND_RETHROW(cusolver::heev_error_handling(working.m_cpu_info(0), 'a'));
    }

    static inline int_type query_worksize(const char JOBZ, const char UPLO, const int_type N, T* A, const int_type LDA, typename get_real_type<T>::type* W, additional_working& /* working */)
    {
        int_type n = N; int_type lda = LDA;
        cusolverEigMode_t jobz; CALL_AND_RETHROW(jobz = get_jobz(JOBZ));
        cublasFillMode_t uplo; CALL_AND_RETHROW(uplo = get_uplo(UPLO));
        int_type worksize;   CALL_AND_RETHROW(cuda_backend::heev_buffersize(jobz, uplo, n, A, lda, W, &worksize);)
        return worksize;
    }

    static inline cusolverEigMode_t get_jobz(const char JOBZ)
    {
        switch(JOBZ)
        {
            case('N') : {return CUSOLVER_EIG_MODE_NOVECTOR;}
            case('V') : {return CUSOLVER_EIG_MODE_VECTOR;}
            default: {RAISE_EXCEPTION("Invalid JOBZ argument.");}
        };
    }

    static inline cublasFillMode_t get_uplo(const char UPLO)
    {
        switch(UPLO)
        {
            case('U') : {return CUBLAS_FILL_MODE_UPPER;}
            case('L') : {return CUBLAS_FILL_MODE_LOWER;}
            default: {RAISE_EXCEPTION("Invalid UPLO argument.");}
        };
    }
};
#endif

template <typename matrix_type>
class hermitian_eigensolver
{
    static_assert(is_hermitian_matrix<matrix_type>::value, "Invalid template parameter for hermitian_eigensolver");
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;
    using helper = hermitian_eigensolver_helper<value_type, backend_type>;
    using int_type = typename helper::int_type;
protected:
    typename helper::additional_working m_additional;
    tensor<value_type, 1, backend_type> m_work;
    tensor<value_type, 2, backend_type> m_mat;
public:
    hermitian_eigensolver() {}
    hermitian_eigensolver(size_type n, bool requires_matrix = true){CALL_AND_HANDLE(resize(n, requires_matrix), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}
    hermitian_eigensolver(const hermitian_eigensolver& o) = default;
    hermitian_eigensolver(hermitian_eigensolver&& o) = default;
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
    hermitian_eigensolver(const mat_type& mat, bool requires_matrix = true){CALL_AND_HANDLE(resize(mat.shape(0), requires_matrix), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}

    /////////////////////////////////////////////////////////////////////////////////////////
    //     Functions for resizing the internal buffers required for computing a symmetric  //
    //                       matrix eigendecomposition.                         //
    /////////////////////////////////////////////////////////////////////////////////////////
    void resize(size_type n, bool requires_matrix = true)
    {
        try
        {
            if(requires_matrix){CALL_AND_HANDLE(m_mat.resize(n, n), "Failed whenn resizing internal matrix.");}  
            CALL_AND_HANDLE(m_additional.resize(n), "Failed whenn resizing additional working space.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize eigensolver object.");
        }
    }
    template <typename mat_type>
    valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type& mat, bool requires_matrix = true){CALL_AND_RETHROW(resize(mat.shape(0), requires_matrix));}
    void resize_workspace(size_type worksize){if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of eigensolver object.");}}

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_mat.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_work.clear(), "Failed when clearing the working array.");
            CALL_AND_HANDLE(m_additional.clear(), "Failed to clear the additional working array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear eigensolver object.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //   Functions for computing the eigendecomposition of a hermitian matrix              //
    /////////////////////////////////////////////////////////////////////////////////////////
    //functions for computing the eigenvalues of a symmetric matrix
    template <typename mat_type, typename vals_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(const mat_type& mat, vals_type& eigs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
            CALL_AND_RETHROW(compute(mat, eigs));
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigenvalues of hermitian matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigenvalues of hermitian matrix.");
        }
    }
    template <typename mat_type, typename vals_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_real_vals_type, vals_type> operator()(mat_type& mat, vals_type& eigs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                CALL_AND_RETHROW(compute(m_mat, eigs));
            }
            else{CALL_AND_RETHROW(compute(mat, eigs));}
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigenvalues of hermitian matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigenvalues of hermitian matrix. ");
        }
    }

    //functions for computing the eigenvalues and right eigenvectors of a symmetric matrix
    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_type, vals_type, vecs_type> operator()(const mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
            CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigendecomposition of hermitian matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigendecomposition of hermitian matrix. ");
        }
    }    
    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_real_vals_vecs_type, vals_type, vecs_type> operator()(mat_type& mat, vals_type& eigs, vecs_type& vecs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to copy input matrix.");
                CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
            }
            else{CALL_AND_RETHROW(compute(mat, eigs, vecs));}
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigendecomposition of hermitian matrix.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eigendecomposition of hermitian matrix.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for determining the optimal workspace for eigendecomposition             //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type, typename vals_type>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_type, vals_type> query_work_space(mat_type& mat, vals_type& eigs)
    {
        char JOBZ = 'N'; char UPLO = 'U'; 
        int_type n = mat.shape(0); int_type lda = mat.shape(1);
        CALL_AND_HANDLE(return helper::query_worksize(JOBZ, UPLO, n, mat.buffer(), lda, eigs.buffer(), m_additional), "Failed to query the optimal workspace for the eigensolver.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_real_vals_vecs_type, vals_type, vecs_type> query_work_space(mat_type& mat, vals_type& eigs, vecs_type& /* vecs */)
    {
        char JOBZ = 'V'; char UPLO = 'U'; 
        int_type n = mat.shape(0); int_type lda = mat.shape(1);
        CALL_AND_HANDLE(return helper::query_worksize(JOBZ, UPLO, n, mat.buffer(), lda, eigs.buffer(), m_additional), "Failed to query the optimal workspace for the eigensolver.");
    }

protected:
    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for computing the eigendecomposition of a symmetric matrix        //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type, typename vals_type>
    void compute(mat_type& mat, vals_type& eigs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, eigs), "Failed to compute eigenvalues of hermitian matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigenvalues of hermitian matrix. Failed to resize the optimal workspace array.");}
        char JOBZ = 'N';    char UPLO = 'U';    int_type n = mat.shape(0);   int_type lda = mat.shape(1); int_type lwork = m_work.size();
        CALL_AND_HANDLE(helper::call(JOBZ, UPLO, n, mat.buffer(), lda, eigs.buffer(), m_work.buffer(), lwork, m_additional), "Failed to compute eigenvalues of hermitian matrix.  Failed when diagonalising matrix.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    void compute(mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, eigs, vecs), "Failed to compute eigendecomposition of hermitian matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigendecomposition of hermitian matrix. Failed to resize the optimal workspace array.");}
        
        char JOBZ = 'V';    char UPLO = 'U';    int_type n = mat.shape(0);   int_type lda = mat.shape(1); int_type lwork = m_work.size();
        CALL_AND_HANDLE(helper::call(JOBZ, UPLO, n, mat.buffer(), lda, eigs.buffer(), m_work.buffer(), lwork, m_additional), "Failed to compute eigendecomposition of hermitian matrix.  Failed when diagonalising matrix.");
        CALL_AND_HANDLE(vecs = adjoint(mat), "Failed to compute eigendecomposition of hermitian matrix.  Failed to compute the adjoint_type of the vecs array to get the correct eigenvectors.");
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
    //  engine for a symmetric matrix                                                      //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type, typename vals_type>
    void validate(const mat_type& mat, vals_type& eigs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of symmetric matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigenvalues of symmetric matrix.  Failed to resize internal storage buffers.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    void validate(const mat_type& mat, vals_type& eigs, vecs_type& vecs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of symmetric matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs), "Failed to compute eigendecomposition of symmetric matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigenvalues of symmetric matrix.  Failed to resize internal storage buffers.");
    }
};
}   //namespace internal


//eigensolver object for hermitian matrix type that is not mutable
template <typename matrix_type>
class eigensolver<matrix_type, typename std::enable_if<is_hermitian_matrix<remove_cvref_t<matrix_type>>::value , void>::type> : public internal::hermitian_eigensolver<remove_cvref_t<matrix_type>>
{
public:
    using base_type = internal::hermitian_eigensolver<remove_cvref_t<matrix_type>>;
    template <typename ... Args>  eigensolver(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};

}   //namespace linalg


#endif //LINALG_DECOMPOSITIONS_EIGENSOLVER_HERMITIAN_MATRIX_HPP//


