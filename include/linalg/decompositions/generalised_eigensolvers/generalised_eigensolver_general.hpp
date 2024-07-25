#ifndef LINALG_DECOMPOSITIONS_EIGENSOLVER_GENERAL_MATRIX_HPP_B
#define LINALG_DECOMPOSITIONS_EIGENSOLVER_GENERAL_MATRIX_HPP_B

#include "generalised_eigensolver_base.hpp"

namespace linalg
{
namespace internal
{

template <typename matrix_type, typename enabler = void> class general_generalised_eigensolver;

//implementation of the general generalised_eigensolver for real valued matrices
template <typename matrix_type>
class general_generalised_eigensolver<matrix_type, typename std::enable_if<!is_complex<typename traits<remove_cvref_t<matrix_type>>::value_type>::value, void>::type>
{
    static_assert(is_dense_matrix<matrix_type>::value, "Invalid template parameter for general_generalised_eigensolver");
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;
    using int_type = typename backend_type::int_type;

protected:
    tensor<value_type, 1, backend_type> m_work;
    tensor<value_type, 2, backend_type> m_A;
    tensor<value_type, 2, backend_type> m_B;
    diagonal_matrix<value_type, backend_type> m_alpha_r;
    diagonal_matrix<value_type, backend_type> m_alpha_i;

public:
    general_generalised_eigensolver() {}
    general_generalised_eigensolver(size_type n, bool requires_matrix = true){CALL_AND_HANDLE(resize(n, requires_matrix), "Failed construct generalised_eigensolver with preallocated size.  Failed when resizing buffers.");}
    template <typename A_type, typename = typename std::enable_if<is_tensor<A_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, A_type, void>>
    general_generalised_eigensolver(const A_type& A, bool requires_matrix = true){CALL_AND_HANDLE(resize(A.shape(0), requires_matrix), "Failed construct generalised_eigensolver with preallocated size.  Failed when resizing buffers.");}

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
                CALL_AND_HANDLE(m_A.resize(n, n), "Failed when resizing internal matrix.");
                CALL_AND_HANDLE(m_B.resize(n, n), "Failed when resizing internal matrix.");
            }  
            CALL_AND_HANDLE(m_alpha_r.resize(n, n), "Failed to resize the internal real part of the eigenvalues array.");
            CALL_AND_HANDLE(m_alpha_i.resize(n, n), "Failed to resize the internal real part of the eigenvalues array.");
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
            CALL_AND_HANDLE(m_A.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_B.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_work.clear(), "Failed when clearing the working array.");
            CALL_AND_HANDLE(m_alpha_r.clear(), "Failed to clear the internal real part of the eigenvalues array.");
            CALL_AND_HANDLE(m_alpha_i.clear(), "Failed to clear the internal real part of the eigenvalues array.");
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
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    valid_decomp_func_2<matrix_type, A_type, B_type, void, validate_complex_vals_type, vals_type> operator()(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_A = A, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
            CALL_AND_HANDLE(m_B = B, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
            CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta));
        }
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating eigenvalues.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigenvalues.");
        }
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    valid_mutable_decomp_func_2<matrix_type, A_type, B_type, void, validate_complex_vals_type, vals_type> operator()(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_A = A, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
                CALL_AND_HANDLE(m_B = B, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
                CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta));
            }
            else{CALL_AND_RETHROW(compute(A, B, alpha, beta));}
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
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, void, validate_complex_vals_vecs_type, vals_type, vecs_type> operator()(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs));
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

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    valid_mutable_decomp_func_2<matrix_type, A_type, B_type, void, validate_complex_vals_vecs_type, vals_type, vecs_type> operator()(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs));
            }
            else
            {
                CALL_AND_HANDLE(A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(A, B, alpha, vecs));
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

    //functions for computing the eigenvalues, and left and right eigenvectors of a general matrix
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, void, validate_complex_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs_r, vecs_l, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs_r, vecs_l));
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

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    valid_mutable_decomp_func_2<matrix_type, A_type, B_type, void, validate_complex_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs_r, vecs_l, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs_r, vecs_l));
            }
            else
            {
                CALL_AND_HANDLE(A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(A, B, alpha, beta, vecs_r, vecs_l));
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
    template <typename A_type, typename B_type, typename vals_type>
    valid_decomp_matrix_type_2<matrix_type, A_type, B_type, size_type> query_work_space(A_type& A, B_type& B, vals_type& beta)
    {
        try
        {
            char JOBVL = 'N'; char JOBVR = 'N';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    value_type worksize;     int_type LWORK = -1;  int_type LDB = B.shape(1);
            CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, m_alpha_r.buffer(), m_alpha_i.buffer(), beta.buffer(), &VL, LDVL, &VR, LDVR, &worksize, LWORK), "Failed to query the optimal worksize for ggev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename A_type, typename B_type, typename vals_type, typename vecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, size_type, validate_complex_vecs_type, vecs_type> query_work_space(A_type& A, B_type& B, vals_type& beta, vecs_type& vecs)
    {
        try
        {
            value_type* rvecs = reinterpret_cast<value_type*>(vecs.buffer());;
            char JOBVL = 'N'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    value_type worksize;     int_type LWORK = -1;    int_type LDB = B.shape(1);
            CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, m_alpha_r.buffer(), m_alpha_i.buffer(), beta.buffer(), &VL, LDVL, rvecs, LDVR, &worksize, LWORK), "Failed to query the optimal worksize for ggev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename A_type, typename B_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, size_type, validate_complex_vecs_rl_type, rvecs_type, lvecs_type> query_work_space(A_type& A, B_type& B, vals_type& beta, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            value_type* rvecsr = reinterpret_cast<value_type*>(vecs_r.buffer()); value_type* rvecsl = reinterpret_cast<value_type*>(vecs_l.buffer());
            char JOBVL = 'V'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    value_type worksize;     int_type LWORK = -1;    int_type LDB = B.shape(1);
            CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, m_alpha_r.buffer(), m_alpha_i.buffer(), beta.buffer(), rvecsl, LDVL, rvecsr, LDVR, &worksize, LWORK), "Failed to query the optimal worksize for ggev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }
protected:
    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for computing the eigendecomposition of a general matrix          //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    void compute(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(A, beta), "Failed to compute eigenvalues of general matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");
        
        //now we make the ggev call to evaluate the eigenvalues
        char JOBVL = 'N'; char JOBVR = 'N';  int_type N = A.shape(0);  int_type LDA = A.shape(1);;   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    int_type LWORK = m_work.size();    int_type LDB = B.shape(1);
        CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, m_alpha_r.buffer(), m_alpha_i.buffer(), beta.buffer(), &VL, LDVL, &VR, LDVR, m_work.buffer(), LWORK), "Failed to compute eigenvalues of general matrix.  Failed when calling ggev.");
        CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_alpha_r.buffer(), 1, m_alpha_i.buffer(), 1, alpha.buffer(), alpha.incx()), "Failed to compute eigenvalues of general matrix.  Failed to interleave real and imaginary part arrays to form complex array.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    void compute(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(A, beta, vecs), "Failed to compute eigenvalues of general matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");
        
        value_type* rvecs = reinterpret_cast<value_type*>(vecs.buffer()); 
        char JOBVL = 'N'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    int_type LWORK = m_work.size();    int_type LDB = B.shape(1);
        CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, m_alpha_r.buffer(), m_alpha_i.buffer(), beta.buffer(), &VL, LDVL, rvecs, LDVR, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of general matrix.  Failed when calling ggev.");
        CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_alpha_r.buffer(), 1, m_alpha_i.buffer(), 1, alpha.buffer(), alpha.incx()), "Failed to compute eigendecomposition of general matrix.  Failed to interleave real and imaginary part arrays to form complex array.");

        CALL_AND_HANDLE(mem_trans::copy(rvecs, N*N, A.buffer()), "Failed to copy the packed eigenvector buffer to the A buffer.");
        CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_alpha_i.buffer(), A.buffer(), vecs.buffer()), "Failed to unpack eigenvectors buffer to complex forA.");
        CALL_AND_HANDLE(vecs = trans(vecs), "Failed to compute eigendecomposition of general matrix.  Failed to construct eigenvector matrix in row major order.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    void compute(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(A, beta, vecs_l, vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");
        
        value_type* rvecsr = reinterpret_cast<value_type*>(vecs_r.buffer()); value_type* rvecsl = reinterpret_cast<value_type*>(vecs_l.buffer());
        char JOBVL = 'V'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);;   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    int_type LWORK = m_work.size();    int_type LDB = B.shape(1);
        CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, m_alpha_r.buffer(), m_alpha_i.buffer(), beta.buffer(), rvecsl, LDVL, rvecsr, LDVR, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of general matrix.  Failed when calling ggev.");
        CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_alpha_r.buffer(), 1, m_alpha_i.buffer(), 1, alpha.buffer(), alpha.incx()), "Failed to compute eigendecomposition of general matrix.  Failed to interleave real and imaginary part arrays to form complex array.");
        
        CALL_AND_HANDLE(mem_trans::copy(rvecsr, N*N, A.buffer()), "Failed to copy the packed eigenvector buffer to the A buffer.");
        CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_alpha_i.buffer(), A.buffer(), vecs_r.buffer()), "Failed to unpack eigenvectors buffer to complex forA.");

        CALL_AND_HANDLE(mem_trans::copy(rvecsl, N*N, A.buffer()), "Failed to copy the packed eigenvector buffer to the A buffer.");
        CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_alpha_i.buffer(), A.buffer(), vecs_l.buffer()), "Failed to unpack eigenvectors buffer to complex forA.");
        //take the inner product of the left and right eigenvectors to allow for a sensible normalisation
        CALL_AND_HANDLE(for(size_type i=0; i<vecs_r.shape(0); ++i){complex<value_type> scaling = static_cast<value_type>(1.0)/dot_product(conj(vecs_l[i]), vecs_r[i]); vecs_r[i] *= scaling;}, "Failed to compute eigendecomposition of general matrix.  Failed to rescale the right eigenvectors.");
        CALL_AND_HANDLE(vecs_r = trans(vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to construct right eigenvectors in row major order.");
        CALL_AND_HANDLE(vecs_l = trans(vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to construct left eigenvectors in row major order.");
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
    //  engine for a general matrix                                                        //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    void validate(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, bool use_matrix)
    {
        ASSERT(A.size(0) == B.size(0) && A.size(1) == B.size(1), "Failed to compute generalised eigenvalues the input arrays are not the same size.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, alpha), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, beta), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(resize(A.shape(0), use_matrix), "Failed to compute eigenvalues of general matrix.  Failed resize buffers.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    void validate(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs, bool use_matrix)
    {
        ASSERT(A.size(0) == B.size(0) && A.size(1) == B.size(1), "Failed to compute generalised eigenvalues the input arrays are not the same size.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, alpha), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, beta), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvectors(A, vecs), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
        CALL_AND_HANDLE(resize(A.shape(0), use_matrix), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    void validate(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l, bool use_matrix)
    {
        ASSERT(A.size(0) == B.size(0) && A.size(1) == B.size(1), "Failed to compute generalised eigenvalues the input arrays are not the same size.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, alpha), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, beta), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvectors(A, vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the right eigenvector matrix.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvectors(A, vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the left eigenvector matrix.");
        CALL_AND_HANDLE(resize(A.shape(0), use_matrix), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }
};

//implementation of the general generalised_eigensolver for complex valued matrices
template <typename matrix_type>
class general_generalised_eigensolver<matrix_type, typename std::enable_if<is_complex<typename traits<remove_cvref_t<matrix_type>>::value_type>::value, void>::type>
{
    static_assert(is_dense_matrix<matrix_type>::value, "Invalid template parameter for general_generalised_eigensolver");
public:
    using value_type = typename traits<matrix_type>::value_type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;
    using int_type = typename backend_type::int_type;
protected:
    tensor<typename get_real_type<value_type>::type, 1, backend_type> m_rwork;
    tensor<value_type, 1, backend_type> m_work;
    tensor<value_type, 2, backend_type> m_A;
    tensor<value_type, 2, backend_type> m_B;
    diagonal_matrix<value_type, backend_type> m_scaling;
public:

    general_generalised_eigensolver() {}
    general_generalised_eigensolver(size_type n, bool requires_matrix = true, bool requires_scaling = false){CALL_AND_HANDLE(resize(n, requires_matrix, requires_scaling), "Failed construct generalised_eigensolver with preallocated size.  Failed when resizing buffers.");}
    template <typename A_type, typename = typename std::enable_if<is_tensor<A_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, A_type, void>>
    general_generalised_eigensolver(const A_type& A, bool requires_matrix = true, bool requires_scaling = false){CALL_AND_HANDLE(resize(A.shape(0), requires_matrix, requires_scaling), "Failed construct generalised_eigensolver with preallocated size.  Failed when resizing buffers.");}

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for resizing the internal buffers required for computing a symmetric     //
    //  tridiagonal matrix eigendecomposition.                                             //
    /////////////////////////////////////////////////////////////////////////////////////////
    void resize(size_type n, bool requires_matrix = true, bool requires_scaling = false)
    {
        try
        {
            if(requires_matrix){CALL_AND_HANDLE(m_A.resize(n, n), "Failed when resizing internal matrix.");}  
            if(requires_matrix){CALL_AND_HANDLE(m_B.resize(n, n), "Failed when resizing internal matrix.");}  
            if(requires_scaling){CALL_AND_HANDLE(m_scaling.resize(n, n), "Failed when resizing the internal scaling matrix required to give sensibly normalised eigenvectors.");}
            CALL_AND_HANDLE(m_rwork.resize(2*n), "Failed when resizing the rwork working space.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize generalised_eigensolver object.");
        }
    }
    template <typename A_type>
    valid_decomp_matrix_type<matrix_type, A_type, void> resize(const A_type& A, bool requires_matrix = true, bool requires_scaling = false){CALL_AND_RETHROW(resize(A.shape(0), requires_matrix, requires_scaling));}
    void resize_workspace(size_type worksize){if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of generalised_eigensolver object.");}}

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_A.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_B.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_work.clear(), "Failed when clearing the working array.");
            CALL_AND_HANDLE(m_rwork.clear(), "Failed when clearing the real working array.");
            CALL_AND_HANDLE(m_scaling.clear(), "Failed when clearing the scaling array.");
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
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    valid_decomp_func_2<matrix_type, A_type, B_type, void, validate_vals_type, vals_type> operator()(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_A = A, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
            CALL_AND_HANDLE(m_B = B, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
            CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigenvalues.");
        }
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    valid_mutable_decomp_func_2<matrix_type, A_type, B_type, void, validate_vals_type, vals_type> operator()(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_A = A, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
                CALL_AND_HANDLE(m_B = B, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
                CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta));
            }
            else{CALL_AND_RETHROW(compute(A, B, alpha, beta));}
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }

    //functions for computing the eigenvalues and right eigenvectors of a general matrix
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, void, validate_vals_vecs_type, vals_type, vecs_type> operator()(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }    
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    valid_mutable_decomp_func_2<matrix_type, A_type, B_type, void, validate_vals_vecs_type, vals_type, vecs_type> operator()(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs));
            }
            else
            {
                CALL_AND_HANDLE(A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(A, B, alpha, beta, vecs));
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }

    //functions for computing the eigenvalues, and left and right eigenvectors of a general matrix
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, void, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs_r, vecs_l, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs_r, vecs_l));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }    

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    valid_mutable_decomp_func_2<matrix_type, A_type, B_type, void, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(A, B, alpha, beta, vecs_r, vecs_l, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(m_B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_A, m_B, alpha, beta, vecs_r, vecs_l));
            }
            else
            {
                CALL_AND_HANDLE(A = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_HANDLE(B = trans(B), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(A, B, alpha, beta, vecs_r, vecs_l));
            }
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
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    valid_decomp_func_2<matrix_type, A_type, B_type, size_type, validate_vals_type, vals_type> query_work_space(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta)
    {
        try
        {
            char JOBVL = 'N'; char JOBVR = 'N';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    value_type worksize;     int_type LWORK = -1;    int_type LDB = B.shape(1);
            CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, alpha.buffer(), beta.buffer(), &VL, LDVL, &VR, LDVR, &worksize, LWORK, m_rwork.buffer()), "Failed to query the optimal worksize for ggev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, size_type, validate_vals_vecs_type, vals_type, vecs_type> query_work_space(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs)
    {
        try
        {
            char JOBVL = 'N'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    value_type worksize;     int_type LWORK = -1;    int_type LDB = B.shape(1);
            CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, alpha.buffer(), beta.buffer(), &VL, LDVL, vecs.buffer(), LDVR, &worksize, LWORK, m_rwork.buffer()), "Failed to query the optimal worksize for ggev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    valid_decomp_func_2<matrix_type, A_type, B_type, size_type, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> query_work_space(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            char JOBVL = 'V'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    value_type worksize;     int_type LWORK = -1;    int_type LDB = B.shape(1);
            CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, alpha.buffer(), beta.buffer(), vecs_l.buffer(), LDVL, vecs_r.buffer(), LDVR, &worksize, LWORK, m_rwork.buffer()), "Failed to query the optimal worksize for ggev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }
protected:
    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for computing the eigendecomposition of a general matrix          //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    void compute(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(A, B, alpha, beta), "Failed to compute eigenvalues of general matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigenvalues of general matrix. Failed to resize the optimal workspace array.");}
        
        char JOBVL = 'N'; char JOBVR = 'N';  int_type N = A.shape(0);  int_type LDA = A.shape(1);;   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    int_type LWORK = m_work.size();    int_type LDB = B.shape(1);
        CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, alpha.buffer(), beta.buffer(), &VL, LDVL, &VR, LDVR, m_work.buffer(), LWORK, m_rwork.buffer()), "Failed to compute eigenvalues of general matrix.  Failed when calling ggev.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    void compute(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(A, B, alpha, beta, vecs), "Failed to compute eigendecomposition of general matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");}

        char JOBVL = 'N'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    int_type LWORK = m_work.size();    int_type LDB = B.shape(1);
        CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, alpha.buffer(), beta.buffer(), &VL, LDVL, vecs.buffer(), LDVR, m_work.buffer(), LWORK, m_rwork.buffer()), "Failed to compute eigenvalues of general matrix.  Failed when calling ggev.");
        
        //now vecs is in column major order so we convert it to row major order
        CALL_AND_HANDLE(vecs = trans(vecs), "Failed to compute eigendecomposition of general matrix.  Failed to construct eigenvectors in row major order.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    void compute(A_type& A, B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(A, B, alpha, beta, vecs_r, vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");}
        
        char JOBVL = 'V'; char JOBVR = 'V';  int_type N = A.shape(0);  int_type LDA = A.shape(1);   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    int_type LWORK = m_work.size();    int_type LDB = B.shape(1);
        CALL_AND_HANDLE(backend_type::ggev(JOBVL, JOBVR, N, A.buffer(), LDA, B.buffer(), LDB, alpha.buffer(), beta.buffer(), vecs_l.buffer(), LDVL, vecs_r.buffer(), LDVR, m_work.buffer(), LWORK, m_rwork.buffer()), "Failed to compute eigenvalues of general matrix.  Failed when calling ggev.");
        
        //take the inner product of the left and right eigenvectors to allow for a sensible normalisation
        using std::sqrt;
        CALL_AND_HANDLE(for(size_type i=0; i<vecs_r.shape(0); ++i){m_scaling[i] = static_cast<value_type>(1.0)/dot_product(conj(vecs_l[i]), vecs_r[i]);}, "Failed to compute eigendecomposition of general matrix.  Failed to compute the rescaling factor.");
        CALL_AND_HANDLE(A = m_scaling*vecs_r, "Failed to compute eigendecomposition of general matrix.  Failed to rescale right eigendecomposition."); 
        CALL_AND_HANDLE(vecs_r = trans(A), "Failed to compute eigendecomposition of general matrix.  Failed to construct right eigenvectors in row major order.");
        CALL_AND_HANDLE(vecs_l = trans(vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to construct left eigenvectors in row major order.");
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
    //  engine for a general matrix                                                        //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename A_type, typename B_type, typename vals_type, typename vals_type2>
    void validate(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta, bool use_matrix)
    {
        ASSERT(A.size(0) == B.size(0) && A.size(1) == B.size(1), "Failed to compute generalised eigenvalues the input arrays are not the same size.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, alpha), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, beta), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(resize(A.shape(0), use_matrix, false), "Failed to compute eigenvalues of general matrix.  Failed resize buffers.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename vecs_type>
    void validate(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta, vecs_type& vecs, bool use_matrix)
    {
        ASSERT(A.size(0) == B.size(0) && A.size(1) == B.size(1), "Failed to compute generalised eigenvalues the input arrays are not the same size.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, alpha), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, beta), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvectors(A, vecs), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
        CALL_AND_HANDLE(resize(A.shape(0), use_matrix, false), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }

    template <typename A_type, typename B_type, typename vals_type, typename vals_type2, typename rvecs_type, typename lvecs_type>
    void validate(const A_type& A, const B_type& B, vals_type& alpha, vals_type2& beta, rvecs_type& vecs_r, lvecs_type& vecs_l, bool use_matrix)
    {
        ASSERT(A.size(0) == B.size(0) && A.size(1) == B.size(1), "Failed to compute generalised eigenvalues the input arrays are not the same size.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, alpha), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvalues(A, beta), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvectors(A, vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the right eigenvector matrix.");
        CALL_AND_HANDLE(internal::generalised_eigensolver_result_validation::eigenvectors(A, vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the left eigenvector matrix.");
        CALL_AND_HANDLE(resize(A.shape(0), use_matrix, true), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }
};
}   //namespace internal

//generalised_eigensolver object for general matrix type that is mutable.  This function provides
//additional evaluating routines that optionally overwrite the input matrix.
//
template <typename matrix_type>
class generalised_eigensolver
<
    matrix_type, 
    typename std::enable_if
    <
        is_dense_matrix<remove_cvref_t<matrix_type>>::value &&
        std::is_same<typename traits<remove_cvref_t<matrix_type>>::backend_type, blas_backend>::value,
        void
    >::type
> : public internal::general_generalised_eigensolver<remove_cvref_t<matrix_type>>
{
public:
    using base_type = internal::general_generalised_eigensolver<remove_cvref_t<matrix_type>>;
    using backend_type = typename traits<remove_cvref_t<matrix_type>>::backend_type;
    using size_type = typename base_type::size_type;
    using value_type = typename base_type::value_type;
    template <typename ... Args>  generalised_eigensolver(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};
}   //namespace linalg


#endif //LINALG_DECOMPOSITIONS_EIGENSOLVER_GENERAL_MATRIX_HPP//



