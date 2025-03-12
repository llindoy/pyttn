#ifndef LINALG_DECOMPOSITIONS_EIGENSOLVER_GENERAL_MATRIX_HPP
#define LINALG_DECOMPOSITIONS_EIGENSOLVER_GENERAL_MATRIX_HPP

#include "eigensolver_base.hpp"

namespace linalg
{
namespace internal
{

template <typename matrix_type, typename enabler = void> class general_eigensolver;

//implementation of the general eigensolver for real valued matrices
template <typename matrix_type>
class general_eigensolver<matrix_type, typename std::enable_if<!is_complex<typename traits<remove_cvref_t<matrix_type>>::value_type>::value, void>::type>
{
    static_assert(is_dense_matrix<matrix_type>::value, "Invalid template parameter for general_eigensolver");
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;
    using int_type = typename backend_type::int_type;

protected:
    tensor<value_type, 1, backend_type> m_work;
    tensor<value_type, 2, backend_type> m_mat;
    diagonal_matrix<value_type, backend_type> m_eigs_r;
    diagonal_matrix<value_type, backend_type> m_eigs_i;

public:
    general_eigensolver() {}
    general_eigensolver(size_type n, bool requires_matrix = true){CALL_AND_HANDLE(resize(n, requires_matrix), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
    general_eigensolver(const mat_type& mat, bool requires_matrix = true){CALL_AND_HANDLE(resize(mat.shape(0), requires_matrix), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for resizing the internal buffers required for computing a symmetric     //
    //  tridiagonal matrix eigendecomposition.                                             //
    /////////////////////////////////////////////////////////////////////////////////////////
    void resize(size_type n, bool requires_matrix = true)
    {
        try
        {
            if(requires_matrix)
            {
                CALL_AND_HANDLE(m_mat.resize(n, n), "Failed when resizing internal matrix.");
            }  
            CALL_AND_HANDLE(m_eigs_r.resize(n, n), "Failed to resize the internal real part of the eigenvalues array.");
            CALL_AND_HANDLE(m_eigs_i.resize(n, n), "Failed to resize the internal real part of the eigenvalues array.");
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
            CALL_AND_HANDLE(m_eigs_r.clear(), "Failed to clear the internal real part of the eigenvalues array.");
            CALL_AND_HANDLE(m_eigs_i.clear(), "Failed to clear the internal real part of the eigenvalues array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear eigensolver object.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for computing the eigendecomposition of a general matrix                 //
    /////////////////////////////////////////////////////////////////////////////////////////
    //functions for computing the eigenvalues of a general matrix
    template <typename mat_type, typename vals_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_complex_vals_type, vals_type> operator()(const mat_type& mat, vals_type& eigs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
            CALL_AND_RETHROW(compute(m_mat, eigs));
        }
        catch(const common::invalid_value& ex)
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

    template <typename mat_type, typename vals_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_complex_vals_type, vals_type> operator()(mat_type& mat, vals_type& eigs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
                CALL_AND_RETHROW(compute(m_mat, eigs));
            }
            else{CALL_AND_RETHROW(compute(mat, eigs));}
        }
        catch(const common::invalid_value& ex)
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
    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_type, vals_type, vecs_type> operator()(const mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
        }
        catch(const common::invalid_value& ex)
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

    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_type, vals_type, vecs_type> operator()(mat_type& mat, vals_type& eigs, vecs_type& vecs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
            }
            else
            {
                CALL_AND_HANDLE(mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(mat, eigs, vecs));
            }
        }
        catch(const common::invalid_value& ex)
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
    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(const mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs_r, vecs_l, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
        }
        catch(const common::invalid_value& ex)
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

    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_complex_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs_r, vecs_l, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
            }
            else
            {
                CALL_AND_HANDLE(mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(mat, eigs, vecs_r, vecs_l));
            }
        }
        catch(const common::invalid_value& ex)
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
    valid_decomp_matrix_type<matrix_type, mat_type, size_type> query_work_space(mat_type& mat)
    {
        try
        {
            char JOBVL = 'N'; char JOBVR = 'N';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    value_type worksize;     int_type LWORK = -1;  
            CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, m_eigs_r.buffer(), m_eigs_i.buffer(), &VL, LDVL, &VR, LDVR, &worksize, LWORK), "Failed to query the optimal worksize for geev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename mat_type, typename vecs_type>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_complex_vecs_type, vecs_type> query_work_space(mat_type& mat, vecs_type& vecs)
    {
        try
        {
            value_type* rvecs = reinterpret_cast<value_type*>(vecs.buffer());;
            char JOBVL = 'N'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    value_type worksize;     int_type LWORK = -1;  
            CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, m_eigs_r.buffer(), m_eigs_i.buffer(), &VL, LDVL, rvecs, LDVR, &worksize, LWORK), "Failed to query the optimal worksize for geev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename mat_type, typename rvecs_type, typename lvecs_type>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_complex_vecs_rl_type, rvecs_type, lvecs_type> query_work_space(mat_type& mat, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            value_type* rvecsr = reinterpret_cast<value_type*>(vecs_r.buffer()); value_type* rvecsl = reinterpret_cast<value_type*>(vecs_l.buffer());
            char JOBVL = 'V'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    value_type worksize;     int_type LWORK = -1;  
            CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, m_eigs_r.buffer(), m_eigs_i.buffer(), rvecsl, LDVL, rvecsr, LDVR, &worksize, LWORK), "Failed to query the optimal worksize for geev call.");
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
    template <typename mat_type, typename vals_type>
    void compute(mat_type& mat, vals_type& eigs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat), "Failed to compute eigenvalues of general matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");
        
        //now we make the geev call to evaluate the eigenvalues
        char JOBVL = 'N'; char JOBVR = 'N';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);;   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    int_type LWORK = m_work.size();  
        CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, m_eigs_r.buffer(), m_eigs_i.buffer(), &VL, LDVL, &VR, LDVR, m_work.buffer(), LWORK), "Failed to compute eigenvalues of general matrix.  Failed when calling geev.");
        CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_eigs_r.buffer(), 1, m_eigs_i.buffer(), 1, eigs.buffer(), eigs.incx()), "Failed to compute eigenvalues of general matrix.  Failed to interleave real and imaginary part arrays to form complex array.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    void compute(mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, vecs), "Failed to compute eigenvalues of general matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");
        
        value_type* rvecs = reinterpret_cast<value_type*>(vecs.buffer()); 
        char JOBVL = 'N'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    int_type LWORK = m_work.size();  
        CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, m_eigs_r.buffer(), m_eigs_i.buffer(), &VL, LDVL, rvecs, LDVR, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of general matrix.  Failed when calling geev.");
        CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_eigs_r.buffer(), 1, m_eigs_i.buffer(), 1, eigs.buffer(), eigs.incx()), "Failed to compute eigendecomposition of general matrix.  Failed to interleave real and imaginary part arrays to form complex array.");

        CALL_AND_HANDLE(mem_trans::copy(rvecs, N*N, mat.buffer()), "Failed to copy the packed eigenvector buffer to the mat buffer.");
        CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_eigs_i.buffer(), mat.buffer(), vecs.buffer()), "Failed to unpack eigenvectors buffer to complex format.");
        CALL_AND_HANDLE(vecs = trans(vecs), "Failed to compute eigendecomposition of general matrix.  Failed to construct eigenvector matrix in row major order.");
    }

    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    void compute(mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, vecs_l, vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to determine optimal work space.");
        CALL_AND_HANDLE(resize_workspace(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");
        
        value_type* rvecsr = reinterpret_cast<value_type*>(vecs_r.buffer()); value_type* rvecsl = reinterpret_cast<value_type*>(vecs_l.buffer());
        char JOBVL = 'V'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);;   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    int_type LWORK = m_work.size();  
        CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, m_eigs_r.buffer(), m_eigs_i.buffer(), rvecsl, LDVL, rvecsr, LDVR, m_work.buffer(), LWORK), "Failed to compute eigendecomposition of general matrix.  Failed when calling geev.");
        CALL_AND_HANDLE(internal::interleave_eigenvalues(N, m_eigs_r.buffer(), 1, m_eigs_i.buffer(), 1, eigs.buffer(), eigs.incx()), "Failed to compute eigendecomposition of general matrix.  Failed to interleave real and imaginary part arrays to form complex array.");
        
        CALL_AND_HANDLE(mem_trans::copy(rvecsr, N*N, mat.buffer()), "Failed to copy the packed eigenvector buffer to the mat buffer.");
        CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_eigs_i.buffer(), mat.buffer(), vecs_r.buffer()), "Failed to unpack eigenvectors buffer to complex format.");

        CALL_AND_HANDLE(mem_trans::copy(rvecsl, N*N, mat.buffer()), "Failed to copy the packed eigenvector buffer to the mat buffer.");
        CALL_AND_HANDLE(internal::unpack_eigenvectors(N, m_eigs_i.buffer(), mat.buffer(), vecs_l.buffer()), "Failed to unpack eigenvectors buffer to complex format.");
        //take the inner product of the left and right eigenvectors to allow for a sensible normalisation
        CALL_AND_HANDLE(for(size_type i=0; i<vecs_r.shape(0); ++i){complex<value_type> scaling = static_cast<value_type>(1.0)/dot_product(conj(vecs_l[i]), vecs_r[i]); vecs_r[i] *= scaling;}, "Failed to compute eigendecomposition of general matrix.  Failed to rescale the right eigenvectors.");
        CALL_AND_HANDLE(vecs_r = trans(vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to construct right eigenvectors in row major order.");
        CALL_AND_HANDLE(vecs_l = trans(vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to construct left eigenvectors in row major order.");
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
    //  engine for a general matrix                                                        //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type, typename vals_type>
    void validate(mat_type& mat, vals_type& eigs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigenvalues of general matrix.  Failed resize buffers.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    void validate(mat_type& mat, vals_type& eigs, vecs_type& vecs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }

    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    void validate(mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the right eigenvector matrix.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the left eigenvector matrix.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }
};

//implementation of the general eigensolver for complex valued matrices
template <typename matrix_type>
class general_eigensolver<matrix_type, typename std::enable_if<is_complex<typename traits<remove_cvref_t<matrix_type>>::value_type>::value, void>::type>
{
    static_assert(is_dense_matrix<matrix_type>::value, "Invalid template parameter for general_eigensolver");
public:
    using value_type = typename traits<matrix_type>::value_type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;
    using int_type = typename backend_type::int_type;
protected:
    tensor<typename get_real_type<value_type>::type, 1, backend_type> m_rwork;
    tensor<value_type, 1, backend_type> m_work;
    tensor<value_type, 2, backend_type> m_mat;
    diagonal_matrix<value_type, backend_type> m_scaling;
public:

    general_eigensolver() {}
    general_eigensolver(size_type n, bool requires_matrix = true, bool requires_scaling = false){CALL_AND_HANDLE(resize(n, requires_matrix, requires_scaling), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}
    template <typename mat_type, typename = typename std::enable_if<is_tensor<mat_type>::value, void>::type, typename = valid_decomp_matrix_type<matrix_type, mat_type, void>>
    general_eigensolver(const mat_type& mat, bool requires_matrix = true, bool requires_scaling = false){CALL_AND_HANDLE(resize(mat.shape(0), requires_matrix, requires_scaling), "Failed construct eigensolver with preallocated size.  Failed when resizing buffers.");}

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for resizing the internal buffers required for computing a symmetric     //
    //  tridiagonal matrix eigendecomposition.                                             //
    /////////////////////////////////////////////////////////////////////////////////////////
    void resize(size_type n, bool requires_matrix = true, bool requires_scaling = false)
    {
        try
        {
            if(requires_matrix){CALL_AND_HANDLE(m_mat.resize(n, n), "Failed when resizing internal matrix.");}  
            if(requires_scaling){CALL_AND_HANDLE(m_scaling.resize(n, n), "Failed when resizing the internal scaling matrix required to give sensibly normalised eigenvectors.");}
            CALL_AND_HANDLE(m_rwork.resize(2*n), "Failed when resizing the rwork working space.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize eigensolver object.");
        }
    }
    template <typename mat_type>
    valid_decomp_matrix_type<matrix_type, mat_type, void> resize(const mat_type& mat, bool requires_matrix = true, bool requires_scaling = false){CALL_AND_RETHROW(resize(mat.shape(0), requires_matrix, requires_scaling));}
    void resize_workspace(size_type worksize){if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to resize workspace of eigensolver object.");}}

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_mat.clear(), "Failed when clearing internal matrix.");
            CALL_AND_HANDLE(m_work.clear(), "Failed when clearing the working array.");
            CALL_AND_HANDLE(m_rwork.clear(), "Failed when clearing the real working array.");
            CALL_AND_HANDLE(m_scaling.clear(), "Failed when clearing the scaling array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear eigensolver object.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Functions for computing the eigendecomposition of a general matrix                 //
    /////////////////////////////////////////////////////////////////////////////////////////
    //functions for computing the eigenvalues of a general matrix
    template <typename mat_type, typename vals_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_vals_type, vals_type> operator()(const mat_type& mat, vals_type& eigs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
            CALL_AND_RETHROW(compute(m_mat, eigs));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigenvalues.");
        }
    }

    template <typename mat_type, typename vals_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_vals_type, vals_type> operator()(mat_type& mat, vals_type& eigs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = mat, "Failed to compute eigenvalues of general matrix.  Failed to copy the matrix into working space.");
                CALL_AND_RETHROW(compute(m_mat, eigs));
            }
            else{CALL_AND_RETHROW(compute(mat, eigs));}
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }

    //functions for computing the eigenvalues and right eigenvectors of a general matrix
    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_type, vals_type, vecs_type> operator()(const mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }    
    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_type, vals_type, vecs_type> operator()(mat_type& mat, vals_type& eigs, vecs_type& vecs, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_mat, eigs, vecs));
            }
            else
            {
                CALL_AND_HANDLE(mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(mat, eigs, vecs));
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }

    //functions for computing the eigenvalues, and left and right eigenvectors of a general matrix
    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    valid_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(const mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs_r, vecs_l, true), "Invalid inputs.");
            CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
            CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate eigendecomposition.");
        }
    }    

    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    valid_mutable_decomp_func<matrix_type, mat_type, void, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> operator()(mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l, bool keep_inputs = true)
    {
        try
        {
            CALL_AND_HANDLE(validate(mat, eigs, vecs_r, vecs_l, keep_inputs), "Invalid inputs.");
            if(keep_inputs)
            {
                CALL_AND_HANDLE(m_mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(m_mat, eigs, vecs_r, vecs_l));
            }
            else
            {
                CALL_AND_HANDLE(mat = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to transpose the matrix so that it is in column major form.");
                CALL_AND_RETHROW(compute(mat, eigs, vecs_r, vecs_l));
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
    template <typename mat_type, typename vals_type>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_type, vals_type> query_work_space(mat_type& mat, vals_type& eigs)
    {
        try
        {
            char JOBVL = 'N'; char JOBVR = 'N';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    value_type worksize;     int_type LWORK = -1;  
            CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, eigs.buffer(), &VL, LDVL, &VR, LDVR, &worksize, LWORK, m_rwork.buffer()), "Failed to query the optimal worksize for geev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_vecs_type, vals_type, vecs_type> query_work_space(mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        try
        {
            char JOBVL = 'N'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    value_type worksize;     int_type LWORK = -1;  
            CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, eigs.buffer(), &VL, LDVL, vecs.buffer(), LDVR, &worksize, LWORK, m_rwork.buffer()), "Failed to query the optimal worksize for geev call.");
            size_type iworksize; CALL_AND_RETHROW(iworksize = internal::worksize_as_integer(worksize));
            return iworksize;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to determine optimal worksize.");
        }
    }

    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    valid_decomp_func<matrix_type, mat_type, size_type, validate_vals_vecs_rl_type, vals_type, rvecs_type, lvecs_type> query_work_space(mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        try
        {
            char JOBVL = 'V'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    value_type worksize;     int_type LWORK = -1;  
            CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, eigs.buffer(), vecs_l.buffer(), LDVL, vecs_r.buffer(), LDVR, &worksize, LWORK, m_rwork.buffer()), "Failed to query the optimal worksize for geev call.");
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
    template <typename mat_type, typename vals_type>
    void compute(mat_type& mat, vals_type& eigs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, eigs), "Failed to compute eigenvalues of general matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigenvalues of general matrix. Failed to resize the optimal workspace array.");}
        
        char JOBVL = 'N'; char JOBVR = 'N';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);;   value_type VL, VR;   int_type LDVL = 1; int_type LDVR=1;    int_type LWORK = m_work.size();  
        CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, eigs.buffer(), &VL, LDVL, &VR, LDVR, m_work.buffer(), LWORK, m_rwork.buffer()), "Failed to compute eigenvalues of general matrix.  Failed when calling geev.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    void compute(mat_type& mat, vals_type& eigs, vecs_type& vecs)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, eigs, vecs), "Failed to compute eigendecomposition of general matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");}

        char JOBVL = 'N'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   value_type VL;   int_type LDVL = 1; int_type LDVR=vecs.shape(1);    int_type LWORK = m_work.size();  
        CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, eigs.buffer(), &VL, LDVL, vecs.buffer(), LDVR, m_work.buffer(), LWORK, m_rwork.buffer()), "Failed to compute eigenvalues of general matrix.  Failed when calling geev.");
        
        //now vecs is in column major order so we convert it to row major order
        CALL_AND_HANDLE(vecs = trans(vecs), "Failed to compute eigendecomposition of general matrix.  Failed to construct eigenvectors in row major order.");
    }

    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    void compute(mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l)
    {
        size_type worksize; CALL_AND_HANDLE(worksize = query_work_space(mat, eigs, vecs_r, vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to determine optimal work space.");
        if(m_work.size() < worksize){CALL_AND_HANDLE(m_work.resize(worksize), "Failed to compute eigendecomposition of general matrix. Failed to resize the optimal workspace array.");}
        
        char JOBVL = 'V'; char JOBVR = 'V';  int_type N = mat.shape(0);  int_type LDA = mat.shape(1);   int_type LDVL = vecs_l.shape(1); int_type LDVR=vecs_r.shape(1);    int_type LWORK = m_work.size();  
        CALL_AND_HANDLE(backend_type::geev(JOBVL, JOBVR, N, mat.buffer(), LDA, eigs.buffer(), vecs_l.buffer(), LDVL, vecs_r.buffer(), LDVR, m_work.buffer(), LWORK, m_rwork.buffer()), "Failed to compute eigenvalues of general matrix.  Failed when calling geev.");
        
        //take the inner product of the left and right eigenvectors to allow for a sensible normalisation
        using std::sqrt;
        CALL_AND_HANDLE(for(size_type i=0; i<vecs_r.shape(0); ++i){m_scaling[i] = static_cast<value_type>(1.0)/dot_product(conj(vecs_l[i]), vecs_r[i]);}, "Failed to compute eigendecomposition of general matrix.  Failed to compute the rescaling factor.");
        CALL_AND_HANDLE(mat = m_scaling*vecs_r, "Failed to compute eigendecomposition of general matrix.  Failed to rescale right eigendecomposition."); 
        CALL_AND_HANDLE(vecs_r = trans(mat), "Failed to compute eigendecomposition of general matrix.  Failed to construct right eigenvectors in row major order.");
        CALL_AND_HANDLE(vecs_l = trans(vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to construct left eigenvectors in row major order.");
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //  Helper Functions for validating inputs and resizing the to eigendecomposition      //
    //  engine for a general matrix                                                        //
    /////////////////////////////////////////////////////////////////////////////////////////
    template <typename mat_type, typename vals_type>
    void validate(const mat_type& mat, vals_type& eigs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix, false), "Failed to compute eigenvalues of general matrix.  Failed resize buffers.");
    }

    template <typename mat_type, typename vals_type, typename vecs_type>
    void validate(const mat_type& mat, vals_type& eigs, vecs_type& vecs, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the eigenvector matrix.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix, false), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }

    template <typename mat_type, typename vals_type, typename rvecs_type, typename lvecs_type>
    void validate(const mat_type& mat, vals_type& eigs, rvecs_type& vecs_r, lvecs_type& vecs_l, bool use_matrix)
    {
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvalues(mat, eigs), "Failed to compute eigenvalues of general matrix.  Failed to ensure the correct shape of the eigenvalues array.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_r), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the right eigenvector matrix.");
        CALL_AND_HANDLE(internal::eigensolver_result_validation::eigenvectors(mat, vecs_l), "Failed to compute eigendecomposition of general matrix.  Failed to ensure the correct shape of the left eigenvector matrix.");
        CALL_AND_HANDLE(resize(mat.shape(0), use_matrix, true), "Failed to compute eigendecomposition of general matrix.  Failed resize buffers.");
    }
};
}   //namespace internal

//eigensolver object for general matrix type that is mutable.  This function provides
//additional evaluating routines that optionally overwrite the input matrix.
template <typename matrix_type>
class eigensolver
<matrix_type, 
    typename std::enable_if
    <
        is_dense_matrix<remove_cvref_t<matrix_type>>::value &&!is_upper_hessenberg_matrix<remove_cvref_t<matrix_type>>::value && !is_hermitian_matrix<remove_cvref_t<matrix_type>>::value &&
        std::is_same<typename traits<remove_cvref_t<matrix_type>>::backend_type, blas_backend>::value,
        void
    >::type
> : public internal::general_eigensolver<remove_cvref_t<matrix_type>>
{
public:
    using base_type = internal::general_eigensolver<remove_cvref_t<matrix_type>>;
    using backend_type = typename traits<remove_cvref_t<matrix_type>>::backend_type;
    using size_type = typename base_type::size_type;
    using value_type = typename base_type::value_type;
    template <typename ... Args>  eigensolver(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};
}   //namespace linalg


#endif //LINALG_DECOMPOSITIONS_EIGENSOLVER_GENERAL_MATRIX_HPP//



