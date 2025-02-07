#ifndef LINALG_BLAS_BACKEND_HPP
#define LINALG_BLAS_BACKEND_HPP

#ifdef USE_MKL
    #define blas_set_num_threads(x) mkl_set_num_threads(x)
#else
    #define blas_set_num_threads(x) 
#endif

#ifdef USE_LIBXSMM
#include <libxsmm.h>
#endif

#include "extended_blas_functions.hpp"
#include "blas_wrapper.hpp"
#include "mkl_wrapper.hpp"
#include "lapack_wrapper.hpp"
#include <random>
#include <vector>
#include <tuple>
#include <algorithm>


//fix up the const correctness here
namespace linalg
{
class blas_backend : public backend_base
{
public:
    using size_type = std::size_t;
    using blas_int_type = blas::blas_int_type;
    using int_type = blas_int_type;
    using index_type = blas_int_type;
#ifndef USE_MKL
    using select_type = bool;
#else
    using select_type = blas_int_type;
#endif
    template <typename T> struct eCop{inline T operator()(const T& a){return a;}};
    template <typename T> struct eCop<complex<T>> {inline complex<T> operator()(const complex<T>& a){return conj(a);}};
    template <typename T> struct eNop{inline T operator()(const T& a){return a;}};

protected:
    static constexpr size_type default_nthreads = 1;
    static constexpr bool default_batchpar = false;

    static size_type& nthreads()
    {
        static size_type _nthreads;
        return _nthreads;
    }

    static bool& batchpar()
    {
        static bool _batchpar;
        return _batchpar;
    }

public:
    static void initialise()
    {
        initialise(default_nthreads, default_batchpar);
        //nthreads() = default_nthreads;
        //batchpar() = default_batchpar;
    }
    static void initialise(size_type _nthreads, bool _batchpar)
    {
#ifdef USE_LIBXSMM
        libxsmm_init(void);
#endif
        set_num_threads(_nthreads); 
        batchpar() = _batchpar;
    }

    static void set_num_threads(size_type _nthreads)
    {
        nthreads() = _nthreads;
        blas_set_num_threads(_nthreads);
    }
    static size_type get_num_threads(){return nthreads();}
    
    static bool is_initialised(){return true;}

    static void destroy()
    {
#ifdef USE_LIBXSMM
        libxsmm_finalize(void);
#endif
    }

public:
    using transform_type = char;
    static constexpr transform_type op_n = 'N';
    static constexpr transform_type op_t = 'T';
    static constexpr transform_type op_h = 'C';
    static constexpr transform_type op_c = 'I';

public:
    template <typename T> static inline bool is_equal(const T* a, const T* b, size_type n){return std::equal(a, a+n, b);}

    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type vector_scalar_product(size_t N, T A, const T* X, size_t INCX, T* Y, size_t INCY)
    {
        if(A != T(1.0)){for(size_type i=0; i<N; ++i){Y[i*INCY] = A*X[i*INCX];}}
        else{for(size_type i=0; i<N; ++i){Y[i*INCY] = X[i*INCX];}}
    }

    //implementations of the various linear algebra operations required.
    template <typename T>
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type axpy(blas_int_type N, T A, const T* X, blas_int_type INCX, T* Y, blas_int_type INCY){blas::axpy(&N, &A, X, &INCX, Y, &INCY);}

    template <typename T>
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type scal(blas_int_type N, T A, T* X, blas_int_type INCX){blas::scal(&N, &A, X, &INCX);}

    template <typename T>
    static inline typename std::enable_if<is_valid_value_type<T>::value, T>::type dot(bool _conj, blas_int_type N, const T* X, blas_int_type INCX, const T* Y, blas_int_type INCY)
    {
        if(!_conj)
        {
            return eblas_kernels::dot(N, X, INCX, Y, INCY);
        }
        else
        {
            return eblas_kernels::conj_dot(N, X, INCX, Y, INCY);
        }
    }

    template <typename T>
    static inline typename std::enable_if<is_valid_value_type<T>::value, T>::type trace(blas_int_type N, const T* X, blas_int_type INCX){return eblas_kernels::trace(N, X, INCX);}

    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    gemv(transform_type TRANSA, blas_int_type M, blas_int_type N, T ALPHA, const T* A, blas_int_type LDA, const T* X, blas_int_type INCX, T BETA, T* Y, blas_int_type INCY)
    {
        blas::gemv(&TRANSA, &M, &N, &ALPHA, A, &LDA, X, &INCX, &BETA, Y, &INCY);
    }

protected:
    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    default_gemm(transform_type TRANSA, transform_type TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, T ALPHA, const T* A, blas_int_type LDA, const T* B, blas_int_type LDB, T BETA, T* C, blas_int_type LDC)
    {
#if defined(USE_MKL) && defined(USE_GEMM3M)
        mkl_extensions::gemm3m(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
#else
        blas::gemm(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
#endif
    }

public:
    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    gemm(transform_type TRANSA, transform_type TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, T ALPHA, const T* A, blas_int_type LDA, const T* B, blas_int_type LDB, T BETA, T* C, blas_int_type LDC)
    {
        //currently we do not bother support libxsmm as it currently doesn't treat complex matrices.  If this is added in the future it will potentially
        //be worth using it, the majority of the matrix operations involved are small matri operations and this could therefore benefit significantly
        //from LIBXSMM
#ifdef USE_LIBXSMM
        //if we are in one of the regimes 
        if(ALPHA == T(1) && BETA == T(0) && TRANSA == op_n)
        {
#ifdef USE_OPENMP
            //libxsmm_gemm_omp()
#else
            //libxsmm_gemm()
#endif
        }
        else
        {
            default_gemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        }
#else
        default_gemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
#endif
    }


protected:
    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    default_batched_gemm(transform_type TRANSA, transform_type TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, T ALPHA, const T* A, 
                                blas_int_type LDA, size_t strideA, const T* B, blas_int_type LDB, size_t strideB, T BETA, T* C, blas_int_type LDC, 
                                size_t strideC, size_t batchCount)
    {
#ifdef USE_MKL
        //here we perform the batched_gemm using the MKL implementation of it.  Due to the fact that we are attempting to keep all of the operations consistent with a column major
        //representation we need to specify that the matrices are in column major order
        mkl_extensions::batched_strided_gemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, strideA, B, LDB, strideB, BETA, C, LDC, strideC, batchCount);
#else
        if(static_cast<blas_int_type>(batchCount) >= K)
        {
#ifdef USE_OPENMP
#ifdef PARALLELISE_EXTENDED_BLAS
            #pragma omp parallel for schedule(static) 
#endif
#endif
            for(size_t i=0; i<batchCount; ++i){gemm(TRANSA, TRANSB, M, N, K, ALPHA, A+i*strideA, LDA, B+i*strideB, LDB, BETA, C+i*strideC, LDC);}
        }
        else
        {
            for(size_t i=0; i<batchCount; ++i){gemm(TRANSA, TRANSB, M, N, K, ALPHA, A+i*strideA, LDA, B+i*strideB, LDB, BETA, C+i*strideC, LDC);}
        }
#endif
    }

public:
    //batched matrix matrix product
    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    batched_gemm(transform_type TRANSA, transform_type TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, T ALPHA, const T* A, 
                                blas_int_type LDA, size_t strideA, const T* B, blas_int_type LDB, size_t strideB, T BETA, T* C, blas_int_type LDC, 
                                size_t strideC, size_t batchCount)
    {   
        //currently we do not bother support libxsmm as it currently doesn't treat complex matrices.  If this is added in the future it will potentially
        //be worth using it, the majority of the matrix operations involved are small matri operations and this could therefore benefit significantly
        //from LIBXSMM
#ifdef USE_LIBXSMM
        //if we are in one of the regimes 
        if(ALPHA == T(1) && BETA == T(0) && TRANSA == op_n)
        {
#ifdef USE_OPENMP
            //libxsmm_gemm_batch_omp()
#else
            //libxsmm_gemm_bath()
#endif
        }
        else
        {
            default_batched_gemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, strideA, B, LDB, strideB, BETA, C, LDC, strideC, batchCount);
        }
#else
        default_batched_gemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, strideA, B, LDB, strideB, BETA, C, LDC, strideC, batchCount);
#endif
    }

    //rank 3-rank 3 to rank 2 tensor contraction implementation
    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type 
    outer_contract(transform_type TRANSA, transform_type TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, T ALPHA, const T* A, 
                                blas_int_type LDA, size_t strideA, const T* B, blas_int_type LDB, size_t strideB, T BETA, T* /*C*/, blas_int_type LDC, 
                                size_t /*strideC*/, size_t batchCount, T* res)
    {   
        for(size_t i=0; i<batchCount; ++i){gemm(TRANSA, TRANSB, M, N, K, ALPHA, A+i*strideA, LDA, B+i*strideB, LDB, BETA, res, LDC);    BETA = 1.0;}
    }

    //complex conjugation of array
    template <typename T> static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type complex_conjugate(blas_int_type /*size*/, const T* const /*X*/, T* const/*Y*/){}
    template <typename T> static inline typename std::enable_if<is_valid_value_type<complex<T>>::value, void>::type complex_conjugate(blas_int_type size, const complex<T>* const X, complex<T>* const Y){for(blas_int_type i=0; i<size; ++i){Y[i] = conj(X[i]);}}

    //function for copying between two dense buffers
    template <typename T> 
    static inline void copy(const T* src, size_type n, T* dest){std::copy_n(src, n, dest);}

    template <typename T> 
    static inline void rank_3_strided_copy(const T* src, size_type n1, size_type n2, size_type n3, T* dest, size_type n4)
    {
        for(size_type i = 0; i < n1; ++i)
        {
            for(size_type j = 0; j < n2; ++j)
            {
                for(size_type k=0; k< n3; ++k)
                {
                    dest[(i*n4+j)*n3+k] = src[(i*n2+j)*n3+k];
                }
            }
        }
    }

    template <typename T> 
    static inline void rank_3_strided_append(const T* pad, size_type n1, size_type n2, size_type n3, size_type iadd, T* dest, size_type n4)
    {
        for(size_type i = 0; i < n1; ++i)
        {
            for(size_type j = 0; j < iadd; ++j)
            {
                size_type aj = j + n2;
                for(size_type k=0; k< n3; ++k)
                {
                    dest[(i*n4+aj)*n3+k] = pad[(j*n1+i)*n3+k];
                }
            }

        }
    }

    template <typename T> 
    static inline void assign(const T* src, size_type n, T* dest, T beta = T(0))
    {
        if(beta == T(0))
        {
            copy(src, n, dest);
        }
        else
        {
            for(size_type i=0; i<n; ++i){dest[i] = beta*dest[i]+ src[i];}
        }
    }

    template <typename T> 
    static inline void assign_real_to_complex(const T* src, size_type n, T* dest, T beta = T(0))
    {
        if(beta == T(0))
        {
            for(size_type i=0; i<n; ++i){dest[i] = complex<T>(src[i], 0.0);}
        }
        else
        {
            for(size_type i=0; i<n; ++i){dest[i] = beta*src[i] + complex<T>(src[i], 0.0);}
        }
    }
    
    template <typename T> 
    static inline void addition_assign(const T* src, size_type n, T* dest){for(size_type i=0; i<n; ++i){dest[i] += src[i];}}

    template <typename T> 
    static inline void subtraction_assign(const T* src, size_type n, T* dest){for(size_type i=0; i<n; ++i){dest[i] -= src[i];}}

    template <typename T> 
    static inline void copy_real_to_complex(const T* src, size_type n, complex<T>* dest){for(size_type i=0; i<n; ++i){dest[i] = complex<T>(src[i], 0.0);}}

    template <typename T> 
    static inline void addition_assign_real_to_complex(const T* src, size_type n, complex<T>* dest){for(size_type i=0; i<n; ++i){dest[i] += complex<T>(src[i], 0.0);}}

    template <typename T> 
    static inline void subtraction_assign_real_to_complex(const T* src, size_type n, complex<T>* dest){for(size_type i=0; i<n; ++i){dest[i] -= complex<T>(src[i], 0.0);}}



    template <typename T> 
    static inline void copy_matrix_subblock(size_type m1, size_type n1, const T* src, size_type lda, T* dest, size_type ldb)
    {   
        for(size_t i=0; i<m1; ++i)
        {
            for(size_t j=0; j<n1; ++j)
            {
                dest[i*ldb+j] = src[i*lda+j];
            }
        }
    }


    template <typename T> 
    static inline void fill_matrix_block(const T* src, size_type m1, size_type n1, T* dest, size_type m2, size_type n2)
    {   
        ASSERT(n1 <= n2 && m1 <= m2, "fill_block call failed.  The subblock is larger than the full matrix.");
        for(size_t i=0; i<m1; ++i)
        {
            for(size_t j=0; j<n1; ++j)
            {
                dest[i*n2+j] = src[i*n1+j];
            }
            for(size_t j=n1; j<n2; ++j)
            {
                dest[i*n2+j] = T(0);
            }
        }
        for(size_t i=m1; i<m2; ++i)
        {
            for(size_t j=0; j<n2; ++j)
            {
                dest[i*n2+j] = T(0);
            }
        }
    }


    template <typename T> 
    static inline void fill_matrix_upper_triangle(const T* src, size_type m1, size_type n1, T* dest, size_type m2, size_type n2)
    {   
        ASSERT(n1 <= n2 && m1 <= m2, "fill_block call failed.  The subblock is larger than the full matrix.");
        for(size_t i=0; i<m1; ++i)
        {
            for(size_t j=0; j<n1; ++j)
            {
                dest[i*n2+j] = src[i*n1+j];
            }
            for(size_t j=n1; j<n2; ++j)
            {
                dest[i*n2+j] = T(0);
            }
        }
        for(size_t i=m1; i<m2; ++i)
        {
            for(size_t j=0; j<n2; ++j)
            {
                dest[i*n2+j] = T(0);
            }
        }
    }

    //functions for copying 
public:
    //all for use of mkl implementation of the sparse matrix multiplications


    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmv(bool conjA, bool conjB, blas_int_type m, blas_int_type n, T1 alpha, const T2* A, blas_int_type inca, const T3* X, blas_int_type incx, T4 beta, T5* Y, blas_int_type incy)
    {
        ASSERT( m >= 0 && n >= 0 && inca >= 0 && incx >= 0 && incy >= 0, "dgmv Call Failed. Input indices must all be positive.");
        if(conjA)
        {
            if(conjB){eblas_kernels::elementwise_multiplication(m, alpha, A, inca, X, incx, beta, Y, incy, eCop<T2>(), eCop<T3>());}
            else{     eblas_kernels::elementwise_multiplication(m, alpha, A, inca, X, incx, beta, Y, incy, eCop<T2>(), eNop<T3>());}            
        }
        else
        {
            if(conjB){eblas_kernels::elementwise_multiplication(m, alpha, A, inca, X, incx, beta, Y, incy, eNop<T2>(), eCop<T3>());}
            else{     eblas_kernels::elementwise_multiplication(m, alpha, A, inca, X, incx, beta, Y, incy, eNop<T2>(), eNop<T3>());}
        }
        //now pad the remaining elements of Y with 0 zeros
        for(size_t i=m; i < static_cast<size_t>(n); ++i){Y[i*incy] = T5(0);}
    }

public:
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmm(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2* A, size_type inca, const T3* B, size_type ldb, T4 beta, T5* C, size_type ldc)
    {   
        if(sizeof(T5) <= sizeof(float))
        {
            dgmm_kernel_selector(6,6,sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
        else if(sizeof(T5) <= sizeof(double))
        {
            dgmm_kernel_selector(5,5,sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
        else
        {
            dgmm_kernel_selector(4,4,sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
    }
private:
    //helper functions for selecting the block size used for the dgmm call if the dense matrix must be transposed to evaluate the kernel.
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmm_kernel_selector(size_t MDMi, size_t MDNi, bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2* A, size_type inca, const T3* B, size_type ldb, T4 beta, T5* C, size_type ldc)
    {   
        bool conjA = false;
        bool conjB = false;
        bool transDense = false;

        size_t TDM = 1 << MDMi;if(m == 1){TDM = 1;}else{while(m < TDM){TDM = TDM >> 1;}}
        size_t TDN = 1 << MDNi;if(n == 1){TDN = 1;}else{while(n < TDN){TDN = TDN >> 1;}}

        if(sparse_left)
        {
            if(opA == op_c || opA == op_h){conjA = true;}
            if(opB == op_c || opB == op_h){conjB = true;}
            if(opB == op_t || opB == op_h){transDense = true;}

            if(!transDense)
            {   
                if(conjA && conjB){         eblas_kernels::dgm_dm_m::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eCop<T2>(), eCop<T3>());}
                else if(conjA && !conjB){   eblas_kernels::dgm_dm_m::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eCop<T2>(), eNop<T3>());}
                else if(!conjA && conjB){   eblas_kernels::dgm_dm_m::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eNop<T2>(), eCop<T3>());}
                else{                       eblas_kernels::dgm_dm_m::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eNop<T2>(), eNop<T3>());}
            }
            else
            {
                if(conjA && conjB){         eblas_kernels::dgm_dm_mt::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eCop<T2>(), eCop<T3>());}
                else if(conjA && !conjB){   eblas_kernels::dgm_dm_mt::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eCop<T2>(), eNop<T3>());}
                else if(!conjA && conjB){   eblas_kernels::dgm_dm_mt::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eNop<T2>(), eCop<T3>());}
                else{                       eblas_kernels::dgm_dm_mt::eval(TDM, TDN, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, eNop<T2>(), eNop<T3>());}
            }
        }
        //if the sparse matrix is on the right then we need to call the dm_dgm_m? kernels
        else
        {
            if(opB == op_c || opB == op_h){conjA = true;}
            if(opA == op_c || opA == op_h){conjB = true;}
            if(opA == op_t || opA == op_h){transDense = true;}

            if(!transDense)
            {
                if(conjA && conjB){         eblas_kernels::dm_dgm_m::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eCop<T3>(), eCop<T2>());}
                else if(conjA && !conjB){   eblas_kernels::dm_dgm_m::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eCop<T3>(), eNop<T2>());}
                else if(!conjA && conjB){   eblas_kernels::dm_dgm_m::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eNop<T3>(), eCop<T2>());}
                else{                       eblas_kernels::dm_dgm_m::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eNop<T3>(), eNop<T2>());}
            }
            else
            {
                if(conjA && conjB){         eblas_kernels::dm_dgm_mt::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eCop<T3>(), eCop<T2>());}
                else if(conjA && !conjB){   eblas_kernels::dm_dgm_mt::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eCop<T3>(), eNop<T2>());}
                else if(!conjA && conjB){   eblas_kernels::dm_dgm_mt::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eNop<T3>(), eCop<T2>());}
                else{                       eblas_kernels::dm_dgm_mt::eval(TDM, TDN, m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, eNop<T3>(), eNop<T2>());}
            }
        }
    }
public:
    template <typename T>
    static inline void csrmv(transform_type opA, bool /* conjB */, blas_int_type m, blas_int_type /*n*/, size_type /*nnz*/, T alpha, const T* A, const int* rowptr, const int* colind, const T* X, blas_int_type incx, T beta, T* Y, blas_int_type incy)
    {
        ASSERT(opA == op_c || opA == op_n, "Failed to compute csrmv.  Transposed csr matrices are currently not supported.");
        eblas_kernels::csrmv::eval(m, alpha, A, rowptr, colind, X, incx, beta, Y, incy, [](const T& a){return a;}, [](const T& a){return a;});
    }

    template <typename T>
    static inline void csrmv(transform_type opA, bool conjB, blas_int_type m, blas_int_type /*n*/, size_type /*nnz*/, complex<T> alpha, const complex<T>* A, const int* rowptr, const int* colind, const complex<T>* X, blas_int_type incx, complex<T> beta, complex<T>* Y, blas_int_type incy)
    {
        ASSERT(opA == op_c || opA == op_n, "Failed to compute csrmv.  Transposed csr matrices are currently not supported.");
        if(opA == op_c)
        {
            if(conjB){eblas_kernels::csrmv::eval(m, alpha, A, rowptr, colind, X, incx, beta, Y, incy, [](const complex<T>& a){return conj(a);}, [](const complex<T>& a){return conj(a);});}
            else{eblas_kernels::csrmv::eval(m, alpha, A, rowptr, colind, X, incx, beta, Y, incy, [](const complex<T>& a){return conj(a);}, [](const complex<T>& a){return a;});}            
        }
        else
        {
            if(conjB){eblas_kernels::csrmv::eval(m, alpha, A, rowptr, colind, X, incx, beta, Y, incy, [](const complex<T>& a){return a;}, [](const complex<T>& a){return conj(a);});}
            else{eblas_kernels::csrmv::eval(m, alpha, A, rowptr, colind, X, incx, beta, Y, incy, [](const complex<T>& a){return a;}, [](const complex<T>& a){return a;});}
        }
    }

public:
    template <typename T>
    static inline void csrmm(bool opres, transform_type opA, transform_type opB, size_type m, size_type n, size_type /*k*/, size_type /*nnz*/, T alpha, const T* A, const int* rowptr, const int* colind, const T* B, size_type ldb, T beta, T* C, size_type ldc)
    {   
        csrmm_kernel_selector<5, 5>(opres, opA, opB, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, [](const T& a){return a;}, [](const T& a){return a;});
    }

    template <typename T>
    static inline void csrmm(bool opres, transform_type opA, transform_type opB, size_type m, size_type n, size_type /*k*/, size_type /*nnz*/, complex<T> alpha, const complex<T>* A, const int* rowptr, const int* colind, const complex<T>* B, size_type ldb, complex<T> beta, complex<T>* C, size_type ldc)
    {   
        csrmm_kernel_selector<4, 4>(opres, opA, opB, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, [](const complex<T>& a){return conj(a);}, [](const complex<T>& a){return a;});
    }


private:
    template <typename Func, bool trans_res>
    struct csrmm_kernel_launcher
    {
        template <typename T, typename COP, typename NOP>
        static inline void launch(size_type TDM, size_type TDN, size_type m, size_type n, T alpha, const T* A, const int* rowptr, const int* colind, const T* B, size_type ldb, T beta, T* C, size_type ldc, COP&& conj_op, NOP&& none_op, bool conjA, bool conjB)
        {
            if(conjA)
            {
                if(conjB){Func::template eval<trans_res>(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<COP>(conj_op), std::forward<COP>(conj_op));}
                else{Func::template eval<trans_res>(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<COP>(conj_op), std::forward<NOP>(none_op));}
            }
            else
            {
                if(conjB){Func::template eval<trans_res>(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<NOP>(none_op), std::forward<COP>(conj_op));}
                else{Func::template eval<trans_res>(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<NOP>(none_op), std::forward<NOP>(none_op));}
            }
        }
    };


    //helper functions for selecting the block size used for the csrmm call if the dense matrix must be transposed to evaluate the kernel.
    template <size_t MDMi, size_t MDNi, typename T, typename COP, typename NOP>
    static inline void csrmm_kernel_selector(bool transRes, transform_type opA, transform_type opB, size_type m, size_type n, T alpha, const T* A, const int* rowptr, const int* colind, const T* B, size_type ldb, T beta, T* C, size_type ldc, COP&& conj_op, NOP&& none_op)
    {   
        ASSERT(opA == op_c || opA == op_n, "csr matrix matrix multiplication kernel cannot handle a transposed csr matrix.");
        bool conjA = (opA == op_c);
        bool conjB = (opB == op_h || opB == op_c);

        bool transDense = (opB == op_h || opB == op_t);
    
        constexpr size_type MAXDIM_M = 1 << MDMi;
        constexpr size_type MAXDIM_N = 1 << MDNi;
        size_type TDM = MAXDIM_M;if(m == 1){TDM = 1;}else{while(m < TDM){TDM = TDM >> 1;}}
        size_type TDN = MAXDIM_N;if(n == 1){TDN = 1;}else{while(n < TDN){TDN = TDN >> 1;}}

        if(transDense)
        {
            using eval_type = eblas_kernels::csr_dm_mt<MAXDIM_M, MAXDIM_N>;
            if(transRes){csrmm_kernel_launcher<eval_type, true>::launch(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<COP>(conj_op), std::forward<NOP>(none_op), conjA, conjB);}
            else{       csrmm_kernel_launcher<eval_type, false>::launch(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<COP>(conj_op), std::forward<NOP>(none_op), conjA, conjB);}
        }
        else
        {
            using eval_type = eblas_kernels::csr_dm_m<MAXDIM_M, MAXDIM_N>;
            if(transRes){csrmm_kernel_launcher<eval_type, true>::launch(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<COP>(conj_op), std::forward<NOP>(none_op), conjA, conjB);}
            else{       csrmm_kernel_launcher<eval_type, false>::launch(TDM, TDN, m, n, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<COP>(conj_op), std::forward<NOP>(none_op), conjA, conjB);}
        }
    }

//#ifndef USE_MKL
private:
    template <blas_int_type MAX_M, blas_int_type MIN_M, typename ... Args>
    static inline void itranspose_kps(blas_int_type m, Args&& ... args)
    {
        if(MAX_M == MIN_M || m > MAX_M){eblas_kernels::itranspose<MAX_M>::eval(m, std::forward<Args>(args)...);;}
        else{itranspose_kps<MAX_M/2, MIN_M>(m, std::forward<Args>(args)...);}
    }
public:
    template <typename T>
    static inline void itranspose(bool _conj, blas_int_type m, const T& alpha, T* mat, const T& beta)
    {
        if(!_conj){itranspose_kps<64, 8>(m, alpha, mat, beta, [](const T& a){return a;});}
        else{itranspose_kps<64, 8>(m, alpha, mat, beta, [](const T& a){return conj(a);});}
    }


private:
    template <blas_int_type M, blas_int_type MAX_N, blas_int_type MIN_N, typename ... Args>
    static inline void transpose_ksps(blas_int_type m, blas_int_type n, Args&& ... args)
    {
        if(MAX_N == MIN_N || n > MAX_N){eblas_kernels::transpose<MAX_N, M>::eval(n, m, std::forward<Args>(args)...);}
        else{transpose_ksps<M, MAX_N/2, MIN_N>(m, n, std::forward<Args>(args)...);}
    }

    //functions for determining the block size to use for the transpose operation
    template <blas_int_type MAX_M, blas_int_type MIN_M, blas_int_type MAX_N, blas_int_type MIN_N, typename ... Args>
    static inline void transpose_kfps(blas_int_type m, blas_int_type n, Args&& ... args)
    {
        if(MAX_M == MIN_M || m > MAX_M){transpose_ksps<MAX_M, MAX_N, MIN_N>(m, n, std::forward<Args>(args)...);}
        else{transpose_kfps<MAX_M/2, MIN_M, MAX_N, MIN_N>(m, n, std::forward<Args>(args)...);}
    }

public:
    //matrix transpose implementations
    template <typename T> 
    static inline void transpose(bool _conj, blas_int_type m, blas_int_type n, const T& alpha, const T* in, const T& beta, T* out)
    {
        if(in == out)
        {
            ASSERT(n == m, "Failed to evaluate blas_backend::transpose.  If an inplace transpose is requested both m and n must be the same.");
            itranspose(_conj, m, alpha, out, beta);
        }
        else
        {
            ASSERT(in != out, "Failed to evaluate blas_backend::transpose.  The input and output buffers must not be the same.");
            if(!_conj){transpose_kfps<64, 8, 64, 8>(m, n, alpha, in, beta, out, [](const T& a){return a;});}
            else{transpose_kfps<64, 8, 64, 8>(m, n, alpha, in, beta, out, [](const T& a){return conj(a);});}
        }
    }

    template <typename T> 
    static inline void batched_transpose(bool _conj, blas_int_type m, blas_int_type n, const T& alpha, const T* in, const T& beta, T* out, blas_int_type batchCount)
    {
        ASSERT(in != out, "Failed to evaluate blas_backend::batched_transpose.  The input and output buffers must not be the same.");

        if(!_conj){transpose_kfps<64, 8, 64, 8>(m, n, batchCount, alpha, in, beta, out, [](const T& a){return a;});}
        else{transpose_kfps<64, 8, 64, 8>(m, n, batchCount, alpha, in, beta, out, [](const T& a){return conj(a);});}
    }

//#else

//#endif

public:    
    template <typename T>
    static inline void transfer_coo_tuple_to_csr(const std::vector<std::tuple<index_type, index_type, T> >& coo, T* vals, index_type* colinds)
    {
        //now we copy the vector buffer to set colinds
        for(size_type i=0; i<coo.size(); ++i)
        {
            colinds[i] = std::get<1>(coo[i]);
            vals[i] = std::get<2>(coo[i]);
        }
    }


    template <typename T>
    static inline void transfer_coo_tuple_to_csc(const std::vector<std::tuple<index_type, index_type, T> >& coo, T* vals, index_type* rowinds)
    {
        //now we copy the vector buffer to set colinds
        for(size_type i=0; i<coo.size(); ++i)
        {
            rowinds[i] = std::get<0>(coo[i]);
            vals[i] = std::get<2>(coo[i]);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Wrappers of the tridiagonalisation routines                                    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
private:
    //general matrices
    static inline void sytrd_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "sytrd call failed.  The argument at the " << -INFO << " position is invalid.";}
            RAISE_EXCEPTION_MESSSTR("Failed to reduce a hermitian matrix to a symmetric tridiagonal form.  ", oss.str());
        }
    }
public:
    static inline void sytrd(const char uplo, const blas_int_type n, float* a, const blas_int_type lda, float* d, float* e, float* tau, float* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; ssytrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);    CALL_AND_RETHROW(sytrd_error_handling(info, 's'));}
    static inline void sytrd(const char uplo, const blas_int_type n, double* a, const blas_int_type lda, double* d, double* e, double* tau, double* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; dsytrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);    CALL_AND_RETHROW(sytrd_error_handling(info, 'd'));}
    static inline void sytrd(const char uplo, const blas_int_type n, complex<float>* a, const blas_int_type lda, float* d, float* e, complex<float>* tau, complex<float>* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; chetrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);    CALL_AND_RETHROW(sytrd_error_handling(info, 'c'));}
    static inline void sytrd(const char uplo, const blas_int_type n, complex<double>* a, const blas_int_type lda, double* d, double* e, complex<double>* tau, complex<double>* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; zhetrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);    CALL_AND_RETHROW(sytrd_error_handling(info, 'z'));}


private:
    //general matrices
    static inline void ormtr_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "ormtr call failed.  The argument at the " << -INFO << " position is invalid.";}
            RAISE_EXCEPTION_MESSSTR("Failed to apply orthogonal matrix stored in elementary reflectors.  ", oss.str());
        }
    }
public:
    static inline void ormtr(const char side, const char uplo, const char trans, const blas_int_type m, const blas_int_type n, float* a, const blas_int_type lda, float* tau, float* c, const blas_int_type ldc, float* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; sormtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);    CALL_AND_RETHROW(ormtr_error_handling(info, 's'));}
    static inline void ormtr(const char side, const char uplo, const char trans, const blas_int_type m, const blas_int_type n, double* a, const blas_int_type lda, double* tau, double* c, const blas_int_type ldc, double* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; dormtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);    CALL_AND_RETHROW(ormtr_error_handling(info, 'd'));}
    static inline void ormtr(const char side, const char uplo, const char trans, const blas_int_type m, const blas_int_type n, complex<float>* a, const blas_int_type lda, complex<float>* tau, complex<float>* c, const blas_int_type ldc, complex<float>* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; cunmtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);    CALL_AND_RETHROW(ormtr_error_handling(info, 'c'));}
    static inline void ormtr(const char side, const char uplo, const char trans, const blas_int_type m, const blas_int_type n, complex<double>* a, const blas_int_type lda, complex<double>* tau, complex<double>* c, const blas_int_type ldc, complex<double>* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; zunmtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);    CALL_AND_RETHROW(ormtr_error_handling(info, 'z'));}


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Wrappers of the lapack generalised eigendecomposition routines                             //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
private:
    //general matrices
    static inline void ggev_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "ggev call failed.  The argument at the " << -INFO << " position is invalid.";}
            else{oss << type << "ggev call failed.  The QR algorithm failed to compute all of the eigenvalues and so no eigenvectors were computed.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the eigenvalues of a general matrix.  ", oss.str());
        }
    }
public:
    static inline void ggev(const char JOBVL, const char JOBVR, const blas_int_type N, float* A, const blas_int_type LDA, float* B, const blas_int_type LDB, float* WR, float* WI, float* beta, float* VL, const blas_int_type LDVL, float* VR, const blas_int_type LDVR, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack; sggev_(&JOBVL, &JOBVR, &N, A, &LDA, B, &LDB, WR, WI, beta, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(ggev_error_handling(INFO, 's'));}
    static inline void ggev(const char JOBVL, const char JOBVR, const blas_int_type N, double* A, const blas_int_type LDA, double* B, const blas_int_type LDB, double* WR, double* WI, double* beta, double* VL, const blas_int_type LDVL, double* VR, const blas_int_type LDVR, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack; dggev_(&JOBVL, &JOBVR, &N, A, &LDA, B, &LDB, WR, WI, beta, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(ggev_error_handling(INFO, 'd'));}
    static inline void ggev(const char JOBVL, const char JOBVR, const blas_int_type N, complex<float>* A, const blas_int_type LDA, complex<float>* B, const blas_int_type LDB, complex<float>* W, complex<float>* beta, complex<float>* VL, const blas_int_type LDVL, complex<float>* VR, const blas_int_type LDVR, complex<float>* WORK, const blas_int_type LWORK, float* RWORK)
    {blas_int_type INFO;   using namespace lapack; cggev_(&JOBVL, &JOBVR, &N, A, &LDA, B, &LDB, W, beta, VL, &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO);    CALL_AND_RETHROW(ggev_error_handling(INFO, 'c'));}
    static inline void ggev(const char JOBVL, const char JOBVR, const blas_int_type N, complex<double>* A, const blas_int_type LDA, complex<double>* B, const blas_int_type LDB, complex<double>* W, complex<double>* beta, complex<double>* VL, const blas_int_type LDVL, complex<double>* VR, const blas_int_type LDVR, complex<double>* WORK, const blas_int_type LWORK, double* RWORK)
    {blas_int_type INFO;   using namespace lapack; zggev_(&JOBVL, &JOBVR, &N, A, &LDA, B, &LDB, W, beta, VL, &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO);    CALL_AND_RETHROW(ggev_error_handling(INFO, 'z'));}

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Wrappers of the lapack eigendecomposition routines                             //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
private:
    //symmetric tridiagonal matrices
    static inline void stev_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            std::string method_name;
            if(type == 's' || type == 'd'){method_name = "stev";}
            else{method_name = "stedc";}
            std::ostringstream oss; 
            if(INFO < 0){oss << type << method_name << " call failed.  The argument at the " << -INFO << " position is invalid.";}
            else{oss << type <<  method_name << " call failed.  The algorithm failed to compute an eigenvalue while working on a submatrix.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the eigenvalues of a symmetric tridiagonal matrix.  ", oss.str());
        }
    }
public:
    static inline void stev(const char COMPZ, const blas_int_type N, float* D, float* E, float* Z, const blas_int_type LDZ, float* WORK){blas_int_type INFO;   using namespace lapack; sstev_(&COMPZ, &N, D, E, Z, &LDZ, WORK, &INFO);     CALL_AND_RETHROW(stev_error_handling(INFO, 's'));}
    static inline void stev(const char COMPZ, const blas_int_type N, double* D, double* E, double* Z, const blas_int_type LDZ, double* WORK){blas_int_type INFO;   using namespace lapack; dstev_(&COMPZ, &N, D, E, Z, &LDZ, WORK, &INFO);     CALL_AND_RETHROW(stev_error_handling(INFO, 'd'));}
//    static inline void stev(const char COMPZ, const blas_int_type N, complex<float>* D, complex<float>* E, complex<float>* Z, const blas_int_type LDZ, complex<float>* WORK, const blas_int_type LWORK, float* RWORK, const blas_int_type LRWORK, int* IWORK, const blas_int_type LIWORK)
//    {blas_int_type INFO;   using namespace lapack; cstedc_(&COMPZ, &N, D, E, Z, &LDZ, WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);     CALL_AND_RETHROW(stev_error_handling(INFO, 'c'));}
//    static inline void stev(const char COMPZ, const blas_int_type N, complex<double>* D, complex<double>* E, complex<double>* Z, const blas_int_type LDZ, complex<double>* WORK, const blas_int_type LWORK, double* RWORK, const blas_int_type LRWORK, int* IWORK, const blas_int_type LIWORK)
//    {blas_int_type INFO;   using namespace lapack; zstedc_(&COMPZ, &N, D, E, Z, &LDZ, WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);     CALL_AND_RETHROW(stev_error_handling(INFO, 'z'));}

private:
    //hermitian matrices
    static inline void heev_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)   
        {
            std::string method_name;
            if(type == 's' || type == 'd'){method_name = "syev";}
            else{method_name = "heev";}
            std::ostringstream oss; 
            if(INFO < 0){oss << type << method_name << "call failed.  The argument at the " << -INFO << " position is invalid.";}
            else{oss << type << method_name << "call failed.  The algorithm failed to converge.  " << INFO << " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the eigenvalues of a hermitian (or real symmetric matrix).  ", oss.str());
        }
    }
public:
    static inline void heev(const char JOBZ, const char UPLO, const blas_int_type N, float* A, const blas_int_type LDA, float* W, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack; ssyev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(heev_error_handling(INFO, 's'));}
    static inline void heev(const char JOBZ, const char UPLO, const blas_int_type N,  double* A, const blas_int_type LDA, double* W, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;  using namespace lapack; dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(heev_error_handling(INFO, 'd'));}
    static inline void heev(const char JOBZ, const char UPLO, const blas_int_type N, complex<float>* A, const blas_int_type LDA, float* W, complex<float>* WORK, const blas_int_type LWORK, float* RWORK)
    {blas_int_type INFO;   using namespace lapack; cheev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, RWORK, &INFO);    CALL_AND_RETHROW(heev_error_handling(INFO, 'c'));}
    static inline void heev(const char JOBZ, const char UPLO, const blas_int_type N, complex<double>* A, const blas_int_type LDA, double* W, complex<double>* WORK, const blas_int_type LWORK, double* RWORK)
    {blas_int_type INFO;   using namespace lapack; zheev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, RWORK, &INFO);    CALL_AND_RETHROW(heev_error_handling(INFO, 'z'));}

private:
    //general matrices
    static inline void geev_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "geev call failed.  The argument at the " << -INFO << " position is invalid.";}
            else{oss << type << "geev call failed.  The QR algorithm failed to compute all of the eigenvalues and so no eigenvectors were computed.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the eigenvalues of a general matrix.  ", oss.str());
        }
    }
public:
    static inline void geev(const char JOBVL, const char JOBVR, const blas_int_type N, float* A, const blas_int_type LDA, float* WR, float* WI, float* VL, const blas_int_type LDVL, float* VR, const blas_int_type LDVR, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack; sgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(geev_error_handling(INFO, 's'));}
    static inline void geev(const char JOBVL, const char JOBVR, const blas_int_type N, double* A, const blas_int_type LDA, double* WR, double* WI, double* VL, const blas_int_type LDVL, double* VR, const blas_int_type LDVR, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack; dgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(geev_error_handling(INFO, 'd'));}
    static inline void geev(const char JOBVL, const char JOBVR, const blas_int_type N, complex<float>* A, const blas_int_type LDA, complex<float>* W, complex<float>* VL, const blas_int_type LDVL, complex<float>* VR, const blas_int_type LDVR, complex<float>* WORK, const blas_int_type LWORK, float* RWORK)
    {blas_int_type INFO;   using namespace lapack; cgeev_(&JOBVL, &JOBVR, &N, A, &LDA, W, VL, &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO);    CALL_AND_RETHROW(geev_error_handling(INFO, 'c'));}
    static inline void geev(const char JOBVL, const char JOBVR, const blas_int_type N, complex<double>* A, const blas_int_type LDA, complex<double>* W, complex<double>* VL, const blas_int_type LDVL, complex<double>* VR, const blas_int_type LDVR, complex<double>* WORK, const blas_int_type LWORK, double* RWORK)
    {blas_int_type INFO;   using namespace lapack; zgeev_(&JOBVL, &JOBVR, &N, A, &LDA, W, VL, &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO);    CALL_AND_RETHROW(geev_error_handling(INFO, 'z'));}

    //upper hessenberg matrices
private:
    //eigenvalues of upper hessenberg matrix
    static inline void hseqr_error_handling(blas_int_type INFO, const char type)
    {    
        if(INFO != 0)   
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "hseqr call failed.  The argument at the " << -INFO << " position is invalid.";}
            else{oss << type << "hseqr call failed to compute all of the eigenvalues.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the eigenvalues of a Hessenberg matrix H.  ", oss.str());
        }
    }
public:
    static inline void hseqr(const char JOB, const char COMPZ, const blas_int_type N, const blas_int_type ILO, const blas_int_type IHI, float* H, const blas_int_type LDH, float* WR, float* WI, float* Z, const blas_int_type LDZ, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;    using namespace lapack; shseqr_(&JOB, &COMPZ, &N, &ILO, &IHI, H, &LDH, WR, WI, Z, &LDZ, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(hseqr_error_handling(INFO, 's'));}
    static inline void hseqr(const char JOB, const char COMPZ, const blas_int_type N, const blas_int_type ILO, const blas_int_type IHI, double* H, const blas_int_type LDH, double* WR, double* WI, double* Z, const blas_int_type LDZ, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;    using namespace lapack; dhseqr_(&JOB, &COMPZ, &N, &ILO, &IHI, H, &LDH, WR, WI, Z, &LDZ, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(hseqr_error_handling(INFO, 'd'));}
    static inline void hseqr(const char JOB, const char COMPZ, const blas_int_type N, const blas_int_type ILO, const blas_int_type IHI, complex<float>* H, const blas_int_type LDH, complex<float>* W, complex<float>* Z, const blas_int_type LDZ, complex<float>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;    using namespace lapack; chseqr_(&JOB, &COMPZ, &N, &ILO, &IHI, H, &LDH, W, Z, &LDZ, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(hseqr_error_handling(INFO, 'c'));}
    static inline void hseqr(const char JOB, const char COMPZ, const blas_int_type N, const blas_int_type ILO, const blas_int_type IHI, complex<double>* H, const blas_int_type LDH, complex<double>* W, complex<double>* Z, const blas_int_type LDZ, complex<double>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;    using namespace lapack; zhseqr_(&JOB, &COMPZ, &N, &ILO, &IHI, H, &LDH, W, Z, &LDZ, WORK, &LWORK, &INFO);    CALL_AND_RETHROW(hseqr_error_handling(INFO, 'z'));}

private:
    //eigenvectors of the upper hessenberg matrix
    static inline void trevc_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            if(INFO < 0)
            {
                std::ostringstream oss; 
                oss << type << "trevc call failed.  The argument at the " << -INFO << " position is invalid.";
                RAISE_EXCEPTION_MESSSTR("Failed to compute eigenvectors of upper quasi-triangular matrix T.  ", oss.str());
            }
            else{RAISE_EXCEPTION("Failed to compute eigenvectors of upper quasi-triangular matrix T.");}
        }
    }

public:
    static inline blas_int_type trevc(const char SIDE, const char HOWMNY, select_type* SELECT, const blas_int_type N, float* T, const blas_int_type LDT, float* VL, const blas_int_type LDVL, float* VR, const blas_int_type LDVR, const blas_int_type MM, float* WORK)
    {blas_int_type INFO, M;     M = 0; using namespace lapack; strevc_(&SIDE, &HOWMNY, SELECT, &N, T, &LDT, VL, &LDVL, VR, &LDVR, &MM, &M, WORK, &INFO);    CALL_AND_RETHROW(trevc_error_handling(INFO, 's'));    return M;}
    static inline blas_int_type trevc(const char SIDE, const char HOWMNY, select_type* SELECT, const blas_int_type N, double* T, const blas_int_type LDT, double* VL, const blas_int_type LDVL, double* VR, const blas_int_type LDVR, const blas_int_type MM, double* WORK)
    {blas_int_type INFO, M;     M = 0;     using namespace lapack; dtrevc_(&SIDE, &HOWMNY, SELECT, &N, T, &LDT, VL, &LDVL, VR, &LDVR, &MM, &M, WORK, &INFO);    CALL_AND_RETHROW(trevc_error_handling(INFO, 'd'));    return M;}
    static inline blas_int_type trevc(const char SIDE, const char HOWMNY, select_type* SELECT, const blas_int_type N, complex<float>* T, const blas_int_type LDT, complex<float>* VL, const blas_int_type LDVL, complex<float>* VR, const blas_int_type LDVR, const blas_int_type MM, complex<float>* WORK, float* RWORK)
    {blas_int_type INFO, M;     M = 0;     using namespace lapack; ctrevc_(&SIDE, &HOWMNY, SELECT, &N, T, &LDT, VL, &LDVL, VR, &LDVR, &MM, &M, WORK, RWORK, &INFO);    CALL_AND_RETHROW(trevc_error_handling(INFO, 'c')); return M;}
    static inline blas_int_type trevc(const char SIDE, const char HOWMNY, select_type* SELECT, const blas_int_type N, complex<double>* T, const blas_int_type LDT, complex<double>* VL, const blas_int_type LDVL, complex<double>* VR, const blas_int_type LDVR, const blas_int_type MM, complex<double>* WORK, double* RWORK)
    {blas_int_type INFO, M;     M = 0;     using namespace lapack; ztrevc_(&SIDE, &HOWMNY, SELECT, &N, T, &LDT, VL, &LDVL, VR, &LDVR, &MM, &M, WORK, RWORK, &INFO);    CALL_AND_RETHROW(trevc_error_handling(INFO, 'z')); return M;}


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                           Wrappers for the lapack lu decomposition routines                            //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static inline void getrf_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO < 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "getrf call failed.  The argument at the " << -INFO << " position is invalid.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
        }
    }

    static void getrf(const blas_int_type M, const blas_int_type N, float* A, const blas_int_type LDA, int* IPIV)
    {blas_int_type info;  using namespace lapack; sgetrf_(&M, &N, A, &LDA, IPIV, &info);      CALL_AND_RETHROW(getrf_error_handling(info, 's'));}
    static void getrf(const blas_int_type M, const blas_int_type N, double* A, const blas_int_type LDA, int* IPIV)
    {blas_int_type info;  using namespace lapack; dgetrf_(&M, &N, A, &LDA, IPIV, &info);      CALL_AND_RETHROW(getrf_error_handling(info, 'd'));}
    static void getrf(const blas_int_type M, const blas_int_type N, complex<float>* A, const blas_int_type LDA, int* IPIV)
    {blas_int_type info;  using namespace lapack; cgetrf_(&M, &N, A, &LDA, IPIV, &info);      CALL_AND_RETHROW(getrf_error_handling(info, 'c'));}
    static void getrf(const blas_int_type M, const blas_int_type N, complex<double>* A, const blas_int_type LDA, int* IPIV)
    {blas_int_type info;  using namespace lapack; zgetrf_(&M, &N, A, &LDA, IPIV, &info);      CALL_AND_RETHROW(getrf_error_handling(info, 'z'));}

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             Wrappers for the lapack linear solver routines                             //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static inline void getrs_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO < 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "getrs call failed.  The argument at the " << -INFO << " position is invalid.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
        }
    }

    static void getrs(const char TRANS, const blas_int_type N, const blas_int_type NRHS, float* A, const blas_int_type LDA, int* IPIV, float* B, const blas_int_type LDB)
    {blas_int_type info;  using namespace lapack; sgetrs_(&TRANS, &N, &NRHS, A, &LDA, IPIV, B, &LDB, &info);      CALL_AND_RETHROW(getrs_error_handling(info, 's'));}
    static void getrs(const char TRANS, const blas_int_type N, const blas_int_type NRHS, double* A, const blas_int_type LDA, int* IPIV, double* B, const blas_int_type LDB)
    {blas_int_type info;  using namespace lapack; dgetrs_(&TRANS, &N, &NRHS, A, &LDA, IPIV, B, &LDB, &info);      CALL_AND_RETHROW(getrs_error_handling(info, 'd'));}
    static void getrs(const char TRANS, const blas_int_type N, const blas_int_type NRHS, complex<float>* A, const blas_int_type LDA, int* IPIV, complex<float>* B, const blas_int_type LDB)
    {blas_int_type info;  using namespace lapack; cgetrs_(&TRANS, &N, &NRHS, A, &LDA, IPIV, B, &LDB, &info);      CALL_AND_RETHROW(getrs_error_handling(info, 'c'));}
    static void getrs(const char TRANS, const blas_int_type N, const blas_int_type NRHS, complex<double>* A, const blas_int_type LDA, int* IPIV, complex<double>* B, const blas_int_type LDB)
    {blas_int_type info;  using namespace lapack; zgetrs_(&TRANS, &N, &NRHS, A, &LDA, IPIV, B, &LDB, &info);      CALL_AND_RETHROW(getrs_error_handling(info, 'z'));}

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                     Wrappers for the lapack singular value decomposition routines                      //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
private:
    static inline void gesvd_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "gesvd call failed.  The argument at the " << -INFO << " position is invalid.";}
            else{oss << type << "gesvd call failed.  DBSQR failed to converge.  " << INFO << " superdiagonal of an intermediate bidiagonal form did not converge to zero.";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
        }
    }

public:
    //singular value decomposition
    static void gesvd(const char jobu, const char jobv, const blas_int_type m, const blas_int_type n, float* A, const blas_int_type lda, float* S, float* U, const blas_int_type ldu, float* vt, const blas_int_type ldvt, float* work, const blas_int_type lwork)
    {blas_int_type info;   using namespace lapack; sgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, &info);  CALL_AND_RETHROW(gesvd_error_handling(info, 's'));}   
    static void gesvd(const char jobu, const char jobv, const blas_int_type m, const blas_int_type n, double* A, const blas_int_type lda, double* S, double* U, const blas_int_type ldu, double* vt, const blas_int_type ldvt, double* work, const blas_int_type lwork)
    {blas_int_type info;  using namespace lapack; dgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, &info);    CALL_AND_RETHROW(gesvd_error_handling(info, 'd'));}   
    static void gesvd(const char jobu, const char jobv, const blas_int_type m, const blas_int_type n, complex<float>* A, const blas_int_type lda, float* S, complex<float>* U, const blas_int_type ldu, complex<float>* vt, const blas_int_type ldvt, complex<float>* work, const blas_int_type lwork, float* rwork)
    {blas_int_type info;   using namespace lapack; cgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, rwork, &info);   CALL_AND_RETHROW(gesvd_error_handling(info, 'c'));}   
    static void gesvd(const char jobu, const char jobv, const blas_int_type m, const blas_int_type n, complex<double>* A, const blas_int_type lda, double* S, complex<double>* U, const blas_int_type ldu, complex<double>* vt, const blas_int_type ldvt, complex<double>* work, const blas_int_type lwork, double* rwork)
    {blas_int_type info;  using namespace lapack; zgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, rwork, &info);   CALL_AND_RETHROW(gesvd_error_handling(info, 'z'));}   

private:
    static inline void gesdd_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            std::ostringstream oss; 
            if(INFO < 0){oss << type << "gesdd call failed.  The argument at the " << -INFO << " position is invalid.";}
            else{oss << type << "gesdd call failed.  DBBSDC failed to converge.  ";}
            RAISE_EXCEPTION_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
        }
    }
public:
    //divide and conquer singular values decomposition
    static void gesdd(const char jobz, const blas_int_type m, const blas_int_type n, float* A, const blas_int_type lda, float* S, float* U, const blas_int_type ldu, float* vt, const blas_int_type ldvt, float* work, const blas_int_type lwork, int* iwork)
    {blas_int_type info;   using namespace lapack; sgesdd_(&jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, iwork, &info);  CALL_AND_RETHROW(gesdd_error_handling(info, 's'));}
    static void gesdd(const char jobz, const blas_int_type m, const blas_int_type n, double* A, const blas_int_type lda, double* S, double* U, const blas_int_type ldu, double* vt, const blas_int_type ldvt, double* work, blas_int_type lwork, int* iwork)
    {blas_int_type info;   using namespace lapack; dgesdd_(&jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, iwork, &info);  CALL_AND_RETHROW(gesdd_error_handling(info, 'd'));}   
    static void gesdd(const char jobz, const blas_int_type m, const blas_int_type n, complex<float>* A, const blas_int_type lda, float* S, complex<float>* U, const blas_int_type ldu, complex<float>* vt, const blas_int_type ldvt, complex<float>* work,const  blas_int_type lwork, float* rwork, int* iwork)
    {blas_int_type info;   using namespace lapack; cgesdd_(&jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);  CALL_AND_RETHROW(gesdd_error_handling(info, 'c'));}   
    static void gesdd(const char jobz, const blas_int_type m, const blas_int_type n, complex<double>* A, const blas_int_type lda, double* S, complex<double>* U, const blas_int_type ldu, complex<double>* vt, const blas_int_type ldvt, complex<double>* work, const blas_int_type lwork, double* rwork, int* iwork)
    {blas_int_type info;   using namespace lapack; zgesdd_(&jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);  CALL_AND_RETHROW(gesdd_error_handling(info, 'z'));}   


    //helper function for the singular values decomposition.  Expands out the vectors resulting from an inplace evaluation
    template <typename T> 
    static inline void expand_svd_vects(blas_int_type nt, T* A, blas_int_type ni, blas_int_type nf, T* work)
    {
        if(ni != nf)
        {
            for(blas_int_type i=0; i<nt; ++i)
            {
                blas_int_type ia = i*ni;  blas_int_type iw = i*nf;
                for(blas_int_type j=0; j<ni; ++j){work[iw + j] = A[ia+j];}
                for(blas_int_type j=ni; j<nf; ++j){work[iw + j] = T(0.0);}
            }
            copy(work, nt*nf, A);
        }
    }

private:
    static inline void geqp3_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            if(INFO < 0)
            {
                std::ostringstream oss; 
                oss << type << "geqp3 call failed.  The argument at the " << -INFO << " position is invalid.";
                RAISE_EXCEPTION_MESSSTR("Failed to compute lq decomposition.  ", oss.str());
            }
            else{RAISE_EXCEPTION("Failed to compute lq decomposition.");}
        }    
    }

public:
    //interface for constructing the qr - here this gives us the lq decomposition as all of our matrices are transposed
    static inline void geqp3(const blas_int_type M, const blas_int_type N, float* A, const blas_int_type LDA, blas_int_type* JPVT, float* TAU, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;   sgeqp3_(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(geqp3_error_handling(INFO, 's'));}
    static inline void geqp3(const blas_int_type M, const blas_int_type N, double* A, const blas_int_type LDA, blas_int_type* JPVT, double* TAU, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;   dgeqp3_(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(geqp3_error_handling(INFO, 'd'));}
    static inline void geqp3(const blas_int_type M, const blas_int_type N, complex<float>* A, const blas_int_type LDA, blas_int_type* JPVT, complex<float>* TAU, complex<float>* WORK, const blas_int_type LWORK, float* RWORK)
    {blas_int_type INFO;   using namespace lapack;    cgeqp3_(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, RWORK, &INFO);   CALL_AND_RETHROW(geqp3_error_handling(INFO, 'c'));}
    static inline void geqp3(const blas_int_type M, const blas_int_type N, complex<double>* A, const blas_int_type LDA, blas_int_type* JPVT, complex<double>* TAU, complex<double>* WORK, const blas_int_type LWORK, double* RWORK)
    {blas_int_type INFO;   using namespace lapack;    zgeqp3_(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, RWORK, &INFO);   CALL_AND_RETHROW(geqp3_error_handling(INFO, 'z'));}


private:
    static inline void geqrf_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            if(INFO < 0)
            {
                std::ostringstream oss; 
                oss << type << "geqrf call failed.  The argument at the " << -INFO << " position is invalid.";
                RAISE_EXCEPTION_MESSSTR("Failed to compute lq decomposition.  ", oss.str());
            }
            else{RAISE_EXCEPTION("Failed to compute lq decomposition.");}
        }    
    }

public:
    //interface for constructing the qr - here this gives us the lq decomposition as all of our matrices are transposed
    static inline void geqrf(const blas_int_type M, const blas_int_type N, float* A, const blas_int_type LDA, float* TAU, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;   sgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(geqrf_error_handling(INFO, 's'));}
    static inline void geqrf(const blas_int_type M, const blas_int_type N, double* A, const blas_int_type LDA, double* TAU, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;   dgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(geqrf_error_handling(INFO, 'd'));}
    static inline void geqrf(const blas_int_type M, const blas_int_type N, complex<float>* A, const blas_int_type LDA, complex<float>* TAU, complex<float>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    cgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(geqrf_error_handling(INFO, 'c'));}
    static inline void geqrf(const blas_int_type M, const blas_int_type N, complex<double>* A, const blas_int_type LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    zgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(geqrf_error_handling(INFO, 'z'));}

private:
    static inline void ungqr_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)   
        {
            if(INFO < 0)
            {
                std::string method_name;
                if(type == 's' || type == 'd'){method_name = "orgqr";}
                else{method_name = "ungqr";}

                std::ostringstream oss; 
                oss << type << " " << method_name << "call failed.  The argument at the " << -INFO << " position is invalid.";
                RAISE_EXCEPTION_MESSSTR("Failed to compute orthogonal Q matrix from elementary reflecators returned by geqp3.  ", oss.str());
            }
            else{RAISE_EXCEPTION("Failed to compute orthogonal Q matrix from elementary reflecators returned by geqp3.");}
        }
    }

public:
    static inline void ungqr(const blas_int_type M, const blas_int_type N, const blas_int_type K, float* A, const blas_int_type LDA, float* TAU, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    sorgqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(ungqr_error_handling(INFO, 's'));}
    static inline void ungqr(const blas_int_type M, const blas_int_type N, const blas_int_type K, double* A, const blas_int_type LDA, double* TAU, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    dorgqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(ungqr_error_handling(INFO, 'd'));}
    static inline void ungqr(const blas_int_type M, const blas_int_type N, const blas_int_type K, complex<float>* A, const blas_int_type LDA, complex<float>* TAU, complex<float>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    cungqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(ungqr_error_handling(INFO, 'c'));}
    static inline void ungqr(const blas_int_type M, const blas_int_type N, const blas_int_type K, complex<double>* A, const blas_int_type LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    zungqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(ungqr_error_handling(INFO, 'z'));}


private:
    static inline void gelqf_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)
        {
            if(INFO < 0)
            {
                std::ostringstream oss; 
                oss << type << "gelqf call failed.  The argument at the " << -INFO << " position is invalid.";
                RAISE_EXCEPTION_MESSSTR("Failed to compute qr decomposition.  ", oss.str());
            }
            else{RAISE_EXCEPTION("Failed to compute qr decomposition.");}
        }    
    }

public:
    //interface for constructing the qr - here this gives us the lq decomposition as all of our matrices are transposed
    static inline void gelqf(const blas_int_type M, const blas_int_type N, float* A, const blas_int_type LDA, float* TAU, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;   sgelqf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(gelqf_error_handling(INFO, 's'));}
    static inline void gelqf(const blas_int_type M, const blas_int_type N, double* A, const blas_int_type LDA, double* TAU, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;   dgelqf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(gelqf_error_handling(INFO, 'd'));}
    static inline void gelqf(const blas_int_type M, const blas_int_type N, complex<float>* A, const blas_int_type LDA, complex<float>* TAU, complex<float>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    cgelqf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(gelqf_error_handling(INFO, 'c'));}
    static inline void gelqf(const blas_int_type M, const blas_int_type N, complex<double>* A, const blas_int_type LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    zgelqf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);   CALL_AND_RETHROW(gelqf_error_handling(INFO, 'z'));}

    static inline void unglq_error_handling(blas_int_type INFO, const char type)
    {
        if(INFO != 0)   
        {
            if(INFO < 0)
            {
                std::string method_name;
                if(type == 's' || type == 'd'){method_name = "orglq";}
                else{method_name = "unglq";}

                std::ostringstream oss; 
                oss << type << " " << method_name << "call failed.  The argument at the " << -INFO << " position is invalid.";
                RAISE_EXCEPTION_MESSSTR("Failed to compute orthogonal Q matrix from elementary reflecators returned by gelqf.  ", oss.str());
            }
            else{RAISE_EXCEPTION("Failed to compute orthogonal Q matrix from elementary reflecators returned by gelqf.");}
        }
    }

public:
    static inline void unglq(const blas_int_type M, const blas_int_type N, const blas_int_type K, float* A, const blas_int_type LDA, float* TAU, float* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    sorglq_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(unglq_error_handling(INFO, 's'));}
    static inline void unglq(const blas_int_type M, const blas_int_type N, const blas_int_type K, double* A, const blas_int_type LDA, double* TAU, double* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    dorglq_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(unglq_error_handling(INFO, 'd'));}
    static inline void unglq(const blas_int_type M, const blas_int_type N, const blas_int_type K, complex<float>* A, const blas_int_type LDA, complex<float>* TAU, complex<float>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    cunglq_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(unglq_error_handling(INFO, 'c'));}
    static inline void unglq(const blas_int_type M, const blas_int_type N, const blas_int_type K, complex<double>* A, const blas_int_type LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type LWORK)
    {blas_int_type INFO;   using namespace lapack;    zunglq_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);     CALL_AND_RETHROW(unglq_error_handling(INFO, 'z'));}


public:
    template <typename T, typename enabled = void> 
    class random_number;

    template <typename T> 
    class random_number<complex<T>, typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
    {
    public:
        template <typename dist, typename rng>
        static inline complex<T> generate_normal(dist& _dist, rng& _rng)
        {
            return complex<T>(_dist(_rng), _dist(_rng));
        }
    };


    template <typename T> 
    class random_number<T, typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
    {
    public:
        template <typename dist, typename rng>
        static inline T generate_normal(dist& _dist, rng& _rng)
        {
            return _dist(_rng);
        }
    };

public:
    template <typename T, typename rng>
    static inline void fill_random_normal(T* t, size_type n, rng& _rng)
    {
        std::uniform_real_distribution<typename get_real_type<T>::type> dist(0, 1);
        for(size_type i = 0; i < n; ++i){t[i] = random_number<T>::generate_normal(dist, _rng);}
    }

public:
    template <typename arr2> 
    static inline void index_to_inds(size_type index, const arr2& strides, std::vector<size_type>& inds)
    {
        for(size_type i = 0; i < inds.size(); ++i)
        {
            inds[i] = index/strides[i];
            index -= inds[i]*strides[i];
        }
    }

    template <typename arr2> 
    static inline size_type inds_to_index(const std::vector<size_type>& inds, const std::vector<size_type>& order, const arr2& strides)
    {
        size_type ind = 0;
        for(size_type i = 0; i < inds.size(); ++i)
        {
            ind += strides[i]*inds[order[i]];
        }
        return ind;
    }

    template <typename T, typename arr1, typename arr2> 
    static inline void tensor_transpose(const T* in, const std::vector<size_type>& inds, const arr1& dims, const arr2& stride, T* out)
    {
        size_type N = inds.size();
        std::vector<size_type> permuted_stride(N);
        permuted_stride[N - 1] = 1;
        for(size_type i = 1; i < N; ++i)
        {
            permuted_stride[N-(i+1)] = permuted_stride[N-i]*dims[inds[N-i]];
        }

        size_t D = stride[0] * dims[0];

        std::vector<size_type> ind(N);
        for(size_type i = 0; i < D; ++i)
        {
            index_to_inds(i, stride, ind);
        
            out[inds_to_index(ind, inds, permuted_stride)] = in[i];
        }
    }
};





}   //namespace linalg

#endif

