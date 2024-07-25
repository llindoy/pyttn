#ifndef LINALG_BACKEND_EXTENDED_BLAS_KERNELS_HPP
#define LINALG_BACKEND_EXTENDED_BLAS_KERNELS_HPP


namespace linalg
{
namespace eblas_kernels
{
////////////////////////////////////////////////////////////////////////////////
//Function for computing the inplace transpose of a square matrix.            //
////////////////////////////////////////////////////////////////////////////////
template <int blockSize>
struct itranspose
{
    template <typename T, typename OPA>
    static inline void eval(int n, const T& alpha, T* A, const T& beta, OPA&& opa)
    {   
        if(beta == T(0.0))
        {
#ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0; i<n; i+=blockSize)
            {
                //deal with the diagonal block
                for(int k=i; k<i+blockSize && k < n; ++k)
                {   
                    //deal with the diagonal element
                    A[k*(n+1)] = alpha*opa(A[k*(n+1)]);
                    //deal with the off diagonal elements
                    for(int l = k+1; l < i+blockSize && l < n; ++l)
                    {
                        T temp1 = alpha*opa(A[l+k*n]);
                        A[l+k*n] = alpha*opa(A[k+l*n]);
                        A[k+l*n] = temp1;
                    }
                }

                //deal with the off diagonal blocks
                for(int j=i+blockSize; j<n; j+= blockSize)
                {
                    //now we perform the transpose within the block.  Note this is currently a really simple version of the matrix transpose
                    //At a later date we might add explicit support for avx intrinsics
                    for(int k = i; k < i+blockSize && k < n; ++k)
                    {
                        for(int l = j; l < j+blockSize && l < n; ++l)
                        {
                            T temp = alpha*opa(A[l+k*n]);
                            A[l+k*n] = alpha*opa(A[k+l*n]);
                            A[k+l*n] = temp;
                        }
                    }
                }
            }
        }
        else
        {
#ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0; i<n; i+=blockSize)
            {
                //deal with the diagonal block
                for(int k=i; k<i+blockSize && k < n; ++k)
                {   
                    //deal with the diagonal element
                    A[k*(n+1)] = alpha*opa(A[k*(n+1)]);
                    //deal with the off diagonal elements
                    for(int l = k+1; l < i+blockSize && l < n; ++l)
                    {
                        T temp1 = alpha*opa(A[l+k*n]);
                        A[l+k*n] = alpha*opa(A[k+l*n])+beta*A[l+k*n];
                        A[k+l*n] = temp1 + beta*A[k+l*n];
                    }
                }

                //deal with the off diagonal blocks
                for(int j=i+blockSize; j<n; j+= blockSize)
                {
                    //now we perform the transpose within the block.  Note this is currently a really simple version of the matrix transpose
                    //At a later date we might add explicit support for avx intrinsics
                    for(int k = i; k < i+blockSize && k < n; ++k)
                    {
                        for(int l = j; l < j+blockSize && l < n; ++l)
                        {
                            T temp = alpha*opa(A[l+k*n]);
                            A[l+k*n] = alpha*opa(A[k+l*n])+beta*A[l+k*n];
                            A[k+l*n] = temp+beta*A[k+l*n];
                        }
                    }
                }
            }
        }
    }
};

template <> struct itranspose<0>{template <typename T, typename OPA> static inline void eval(int /* n */, const T& /* alpha */, T* /* A */, const T& /* beta */, OPA&& /* opa */){RAISE_EXCEPTION("Should never get here");}};

////////////////////////////////////////////////////////////////////////////////
//Function for computing the transpose of a general rectangular matrix.       //
////////////////////////////////////////////////////////////////////////////////
template <int blockSizeM, int blockSizeN>
struct transpose
{
    template <typename T, typename OPA>
    static inline void eval(int m, int n, const T& alpha, const T* linalg_restrict A, const T& beta, T* linalg_restrict B, OPA&& opa)
    {   
        if(beta == T(0.0))
        {
#ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0; i<n; i+=blockSizeN)
            {
                for(int j=0; j<m; j+= blockSizeM)
                {
                    //now we perform the transpose within the block.  Note this is currently a really simple version of the matrix transpose
                    //At a later date we might add explicit support for avx intrinsics
                    for(int k = i; k < i+blockSizeN && k < n; ++k)
                    {
                        for(int l = j; l < j+blockSizeM && l < m; ++l)
                        {
                            B[k+l*n] = alpha*opa(A[l+k*m]);
                        }
                    }
                }
            }
        }
        else
        {
#ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0; i<n; i+=blockSizeN)
            {
                for(int j=0; j<m; j+= blockSizeM)
                {
                    //now we perform the transpose within the block.  Note this is currently a really simple version of the matrix transpose
                    //At a later date we might add explicit support for avx intrinsics
                    for(int k = i; k < i+blockSizeN && k < n; ++k)
                    {
                        for(int l = j; l < j+blockSizeM && l < m; ++l)
                        {
                            B[k+l*n] = alpha*opa(A[l+k*m]) + beta*B[k+l*n];
                        }
                    }
                }
            }
        }
    }

    //function implementing a blocked/batched transpose
    template <typename T, typename OPA>
    static inline void eval(int m, int n, int batchCount, const T& alpha, const T* linalg_restrict A, const T& beta, T* linalg_restrict B, OPA&& opa)
    {
        //iterate over the different blocks of the matrix
        //n is the slower index as we are dealing with a column major matrix
        int skip = m*n;
        if( beta == T(0.0))
        {
#ifdef USE_OPENMP
            #pragma omp parallel for collapse(3) schedule(static)
#endif
            for(int batch=0; batch < batchCount; ++batch)
            {
                for(int i=0; i<n; i+=blockSizeN)
                {
                    for(int j=0; j<m; j+= blockSizeM)
                    {
                        int bskip = batch*skip;
                        //now we perform the transpose within the block.  Note this is currently a really simple version of the matrix transpose
                        //At a later date we might add explicit support for avx intrinsics
                        for(int k = i; k < i+blockSizeN && k < n; ++k){for(int l = j; l < j+blockSizeM && l < m; ++l){B[bskip+k+l*n] = alpha*opa(A[bskip+l+k*m]);}}
                    }
                }
            }
        }
        else
        {
#ifdef USE_OPENMP
            #pragma omp parallel for collapse(3) schedule(static)
#endif
            for(int batch=0; batch < batchCount; ++batch)
            {
                for(int i=0; i<n; i+=blockSizeN)
                {
                    for(int j=0; j<m; j+= blockSizeM)
                    {
                        int bskip = batch*skip;
                        //now we perform the transpose within the block.  Note this is currently a really simple version of the matrix transpose
                        //At a later date we might add explicit support for avx intrinsics
                        for(int k = i; k < i+blockSizeN && k < n; ++k){for(int l = j; l < j+blockSizeM && l < m; ++l){B[bskip+k+l*n] = alpha*opa(A[bskip+l+k*m])+B[bskip+k+l*n]*beta;}}
                    }
                }
            }
        }
    }
};

template <int blockSizeM>
struct transpose<blockSizeM, 0>
{
    template <typename T, typename OPA>static inline void eval(int /* m */, int /* n */, const T& /* alpha */, const T* linalg_restrict /* A */, const T& /* beta */, T* linalg_restrict /* B */, OPA&& /* opa */){RAISE_EXCEPTION("Should never get here");}
    template <typename T, typename OPA> static inline void eval(int /* m */, int /* n */, int /* batchCount */, const T& /* alpha */, const T* linalg_restrict /* A */, const T& /* beta */, T* linalg_restrict /* B */, OPA&& /* opa */){RAISE_EXCEPTION("Should never get here");}
};

template <int blockSizeN>
struct transpose<0, blockSizeN>
{
    template <typename T, typename OPA>static inline void eval(int /* m */, int /* n */, const T& /* alpha */, const T* linalg_restrict /* A */, const T& /* beta */, T* linalg_restrict /* B */, OPA&& /* opa */){RAISE_EXCEPTION("Should never get here");}
    template <typename T, typename OPA> static inline void eval(int /* m */, int /* n */, int /* batchCount */, const T& /* alpha */, const T* linalg_restrict /* A */, const T& /* beta */, T* linalg_restrict /* B */, OPA&& /* opa */){RAISE_EXCEPTION("Should never get here");}
};

template <>
struct transpose<0, 0>
{
    template <typename T, typename OPA>static inline void eval(int /* m */, int /* n */, const T& /* alpha */, const T* linalg_restrict /* A */, const T& /* beta */, T* linalg_restrict /* B */, OPA&& /* opa */){RAISE_EXCEPTION("Should never get here");}
    template <typename T, typename OPA> static inline void eval(int /* m */, int /* n */, int /* batchCount */, const T& /* alpha */, const T* linalg_restrict /* A */, const T& /* beta */, T* linalg_restrict /* B */, OPA&& /* opa */){RAISE_EXCEPTION("Should never get here");}
};



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//Functions implementing diagonal matrix - dense matrix multiplication                                  //
//at a later date it might be worth implementing these kernels using rectangular tiles to               //
//allow better handling of rectangular matrices (as are often present in the ttns integrator).          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//diagonal matrix * dense matrix.  Both C and B can be accessed in a contiguous fashion here so there is no need to do anything fancy.
class dgm_dm_m
{
protected:
    template <typename FuncA, typename FuncCalc1, typename FuncCalc2, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void inner_loop(const size_t m, const size_t n, const size_t k, T1 alpha, const T2* A, const size_t inca, const T3* B, const size_t ldb, T4 beta, T5* C, const size_t ldc, OPA&& opa, OPB&& opb, FuncA&& fa, FuncCalc1&& fcalc1, FuncCalc2&& fcalc2)
    {
        size_t min_km = m > k ? k : m;
#ifdef USE_OPENMP
        #pragma omp parallel for  schedule(static)
#endif
        for(size_t j=0; j<min_km; ++j)
        {
            auto opaj = fa(A, j*inca, alpha, std::forward<OPA>(opa));
            size_t bind = ldb*j; size_t cind = ldc*j;
            for(size_t i=0; i<n; ++i){C[cind+i] = fcalc1(C, opaj, B, beta, i, cind, bind, std::forward<OPB>(opb));}
        }
#ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(size_t j=min_km; j<m; ++j){size_t cind = ldc*j;    for(size_t i=0; i<n; ++i){C[cind+i] = fcalc2(C, beta, i, cind);}}
    }

public:
    template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void eval(const size_t /* TDM */, const size_t /* TDN */, const size_t _m, const size_t _n, const size_t _k, T1 _alpha, const T2* _A, const size_t _inca, const T3* _B, const size_t _ldb, T4 _beta, T5* _C, const size_t _ldc, OPA&& _opa, OPB&& _opb)
    {
        if(_beta != T4(0.0))
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opa(T2()));
                inner_loop(_m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T2* A, size_t j, const T1& alpha, OPA&& opa){return alpha*opa(A[j]);},
                    [](const T5* C, const T6& opaj, const T3* B, const T4& beta, size_t i, size_t cind, size_t bind, OPB&& opb){return beta*C[cind+i]+opaj*opb(B[bind+i]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
            }
            else
            {
                using T6 = T2;
                inner_loop(_m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T2* A, size_t j, const T1& /*alpha*/, OPA&& opa){return opa(A[j]);},
                    [](const T5* C, const T6& opaj, const T3* B, const T4& beta, size_t i, size_t cind, size_t bind, OPB&& opb){return beta*C[cind+i]+opaj*opb(B[bind+i]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
            }
        }
        else
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opa(T2()));
                inner_loop(_m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T2* A, size_t j, const T1& alpha, OPA&& opa){return alpha*opa(A[j]);},
                    [](const T5* /*C*/, const T6& opaj, const T3* B, const T4& /*beta*/, size_t i, size_t /*cind*/, size_t bind, OPB&& opb){return opaj*opb(B[bind+i]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
            else
            {
                using T6 = T2;
                inner_loop(_m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T2* A, size_t j, const T1& /*alpha*/, OPA&& opa){return opa(A[j]);},
                    [](const T5* /*C*/, const T6& opaj, const T3* B, const T4& /*beta*/, size_t i, size_t /*cind*/, size_t bind, OPB&& opb){return opaj*opb(B[bind+i]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
        }
    }
};  //dgm_dm_m


//diagonal matrix * transpose(dense matrix).  One of C and B must be accessed non-contiguously so we use a blocking algorithm
class dgm_dm_mt
{
protected:
    template <typename FuncA, typename FuncCalc1, typename FuncCalc2, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void inner_loop(const size_t TDM, const size_t TDN, const size_t m, const size_t n, const size_t k, T1 alpha, const T2* A, const size_t inca, const T3* B, const size_t ldb, T4 beta, T5* C, const size_t ldc, OPA&& opa, OPB&& opb, FuncA&& fa, FuncCalc1&& fcalc1, FuncCalc2&& fcalc2)
    {
        size_t min_km = m > k ? k : m;
#ifdef USE_OPENMP
        #pragma omp parallel  for schedule(static)
#endif
        for(size_t jb=0; jb<m; jb+=TDM)
        {
            if(jb < min_km && jb + TDM < min_km) 
            {
                for(size_t ib=0; ib<n; ib+=TDN)
                {
                    for(size_t j=jb; j < jb+TDM; ++j)
                    {
                        auto opaj = fa(A, j*inca, alpha, std::forward<OPA>(opa));
                        size_t cind = ldc*j;
                        for(size_t i=ib; i<ib+TDN && i < n; ++i){size_t bind = i*ldb;    C[cind+i] = fcalc1(C, opaj, B, beta, j, i, cind, bind, std::forward<OPB>(opb));}
                    }
                }
            }
            else if(jb > min_km)
            {
                for(size_t ib=0; ib<n; ib+=TDN)
                {
                    for(size_t j=jb; j < jb+TDM && j < m; ++j)
                    {
                        size_t cind = ldc*j; for(size_t i=ib; i<ib+TDN && i < n; ++i){C[cind+i] = fcalc2(C, beta, i, cind);}
                    }
                }
            }
            else
            {
                for(size_t ib=0; ib<n; ib+=TDN)
                {
                    for(size_t j=jb; j < jb+TDM; ++j)
                    {
                        size_t cind = ldc*j;
                        if(j < min_km)
                        {
                            auto opaj = fa(A, j*inca, alpha, std::forward<OPA>(opa));
                            for(size_t i=ib; i<ib+TDN && i < n; ++i)
                            {
                                size_t bind = i*ldb;    C[cind+i] = fcalc1(C, opaj, B, beta, j, i, cind, bind, std::forward<OPB>(opb));
                            }
                        }
                        else if(j<m){for(size_t i=ib; i<ib+TDN && i < n; ++i){C[cind+i] = fcalc2(C, beta, i, cind);}}
                    }
                }
            }
        }
    }
public:
    template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void eval(const size_t TDM, const size_t TDN, const size_t _m, const size_t _n, const size_t _k, T1 _alpha, const T2* _A, const size_t _inca, const T3* _B, const size_t _ldb, T4 _beta, T5* _C, const size_t _ldc, OPA&& _opa, OPB&& _opb)
    {
        if(_beta != T4(0.0))
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opa(T2()));
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb),
                    [](const T2* A, size_t j, const T1& alpha, OPA&& opa){return alpha*opa(A[j]);},
                    [](const T5* C, const T6& opaj, const T3* B, const T4& beta, size_t j, size_t i, size_t cind, size_t bind, OPB&& opb){return beta*C[cind+i]+opaj*opb(B[bind+j]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
            }
            else
            {
                using T6 = T2;
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb),
                    [](const T2* A, size_t j, const T1& /*alpha*/, OPA&& opa){return opa(A[j]);},
                    [](const T5* C, const T6& opaj, const T3* B, const T4& beta, size_t j, size_t i, size_t cind, size_t bind, OPB&& opb){return beta*C[cind+i]+opaj*opb(B[bind+j]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
            }
        }
        else
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opa(T2()));
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb),
                    [](const T2* A, size_t j, const T1& alpha, OPA&& opa){return alpha*opa(A[j]);},
                    [](const T5* /*C*/, const T6& opaj, const T3* B, const T4& /*beta*/, size_t j, size_t /*i*/, size_t /*cind*/, size_t bind, OPB&& opb){return opaj*opb(B[bind+j]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
            else
            {
                using T6 = T2;
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _inca, _B, _ldb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb),
                    [](const T2* A, size_t j, const T1& /*alpha*/, OPA&& opa){return opa(A[j]);},
                    [](const T5* /*C*/, const T6& opaj, const T3* B, const T4& /*beta*/, size_t j, size_t /*i*/, size_t /*cind*/, size_t bind, OPB&& opb){return opaj*opb(B[bind+j]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
        }
    }
};  //dgm_dm_mt


//dense matrix * diagonal matrix
class dm_dgm_m
{
protected:
    template <typename FuncA, typename FuncCalc1, typename FuncCalc2, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void inner_loop(const size_t m, const size_t n, const size_t k, T1 alpha, const T2* A, const size_t lda, const T3* B, const size_t incb, T4 beta, T5* C, const size_t ldc, OPA&& opa, OPB&& opb, FuncA&& fa, FuncCalc1&& fcalc1, FuncCalc2&& fcalc2)
    {
        size_t min_kn = n > k ? k : n;
#ifdef USE_OPENMP
        #pragma omp parallel for  schedule(static)
#endif
        for(size_t j=0; j<m; ++j)
        {
            size_t aind = lda*j; size_t cind = ldc*j;
            for(size_t i=0; i<min_kn; ++i)
            {
                auto opbi = fa(B, i*incb, alpha, std::forward<OPB>(opb));
                C[cind+i] = fcalc1(C, opbi, A, beta, i, cind, aind, std::forward<OPA>(opa));
            }
            for(size_t i=min_kn; i<n; ++i){C[cind+i] = fcalc2(C, beta, i, cind);}
        }
    }
public:
    template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void eval(const size_t /* TDM */, const size_t /* TDN */, const size_t _m, const size_t _n, const size_t _k, T1 _alpha, const T2* _A, const size_t _lda, const T3* _B, const size_t _incb, T4 _beta, T5* _C, const size_t _ldc, OPA&& _opa, OPB&& _opb)
    {
        if(_beta != T4(0.0))
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opb(T3()));
                inner_loop(_m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& alpha, OPB&& opb){return alpha*opb(B[j]);},
                    [](const T5* C, const T6& opbj, const T2* A, const T4& beta, size_t i, size_t cind, size_t aind, OPA&& opa){return beta*C[cind+i]+opbj*opa(A[aind+i]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
            }
            else
            {
                using T6 = T3;
                inner_loop(_m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& /*alpha*/, OPB&& opb){return opb(B[j]);},
                    [](const T5* C, const T6& opbj, const T2* A, const T4& beta, size_t i, size_t cind, size_t aind, OPA&& opa){return beta*C[cind+i]+opbj*opa(A[aind+i]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
           }
        }
        else
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opb(T3()));
                inner_loop(_m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& alpha, OPB&& opb){return alpha*opb(B[j]);},
                    [](const T5* /*C*/, const T6& opbj, const T2* A, const T4& /*beta*/, size_t i, size_t /*cind*/, size_t aind, OPA&& opa){return opbj*opa(A[aind+i]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
            else
            {
                using T6 = T3;
                inner_loop(_m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& /* alpha */, OPB&& opb){return opb(B[j]);},
                    [](const T5* /*C*/, const T6& opbj, const T2* A, const T4& /*beta*/, size_t i, size_t /*cind*/, size_t aind, OPA&& opa){return opbj*opa(A[aind+i]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
        }
    }
};  //dm_dgm_m


//transpose(dense matrix) * diagonal matrix
class dm_dgm_mt
{
    template <typename FuncA, typename FuncCalc1, typename FuncCalc2, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void inner_loop(const size_t TDM, const size_t TDN, const size_t m, const size_t n, const size_t k, T1 alpha, const T2* A, const size_t lda, const T3* B, const size_t incb, T4 beta, T5* C, const size_t ldc, OPA&& opa, OPB&& opb, FuncA&& fa, FuncCalc1&& fcalc1, FuncCalc2&& fcalc2)
    {
        size_t min_kn = n > k ? k : n;
#ifdef USE_OPENMP
        #pragma omp parallel  for schedule(static)
#endif
        for(size_t jb=0; jb<m; jb+=TDM)
        {
            for(size_t ib=0; ib<n; ib+=TDN)
            {
                if(ib < min_kn && ib+TDM < min_kn)
                {
                    for(size_t j=jb; j < jb+TDN && j<m; ++j)
                    {
                        size_t cind = ldc*j;
                        for(size_t i=ib; i<ib+TDM; ++i)
                        {
                            auto opbi = fa(B, i*incb, alpha, std::forward<OPB>(opb));
                            size_t aind = i*lda;    
                            C[cind+i] = fcalc1(C, opbi, A, beta, j, i, cind, aind, std::forward<OPA>(opa));
                        }
                    }
                }
                else if(ib > min_kn)
                {
                    for(size_t j=jb; j < jb+TDN && j<m; ++j)
                    {
                        size_t cind = ldc*j;
                        for(size_t i=ib; i<ib+TDM && i < n; ++i){C[cind+i] = fcalc2(C, beta, i, cind);}
                    }
                }
                else
                {
                    for(size_t j=jb; j < jb+TDN && j<m; ++j)
                    {
                        size_t cind = ldc*j;
                        for(size_t i=ib; i<min_kn; ++i)
                        {
                            auto opbi = fa(B, i*incb, alpha, std::forward<OPB>(opb));
                            size_t aind = i*lda;    
                            C[cind+i] = fcalc1(C, opbi, A, beta, j, i, cind, aind, std::forward<OPA>(opa));
                        }
                        for(size_t i=min_kn; i<ib+TDM && i < n; ++i){C[cind+i] = fcalc2(C, beta, i, cind);}
                    }
                }
            }
        }
    }
public:
    template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void eval(const size_t TDM, const size_t TDN, const size_t _m, const size_t _n, const size_t _k, T1 _alpha, const T2* _A, const size_t _lda, const T3* _B, const size_t _incb, T4 _beta, T5* _C, const size_t _ldc, OPA&& _opa, OPB&& _opb)
    {
        if(_beta != T4(0.0))
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opb(T3()));
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& alpha, OPB&& opb){return alpha*opb(B[j]);},
                    [](const T5* C, const T6& opbj, const T2* A, const T4& beta, size_t j, size_t i, size_t cind, size_t aind, OPA&& opa){return beta*C[cind+i]+opbj*opa(A[aind+j]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
            }
            else
            {
                using T6 = T3;
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& /* alpha */, OPB&& opb){return opb(B[j]);},
                    [](const T5* C, const T6& opbj, const T2* A, const T4& beta, size_t j, size_t i, size_t cind, size_t aind, OPA&& opa){return beta*C[cind+i]+opbj*opa(A[aind+j]);},
                    [](const T5* C,  const T4& beta, size_t i, size_t cind){return C[cind+i]*beta;});
           }
        }
        else
        {
            if(_alpha != T1(1.0))
            {
                using T6 = decltype(T1()*_opb(T3()));
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& alpha, OPB&& opb){return alpha*opb(B[j]);},
                    [](const T5* /*C*/, const T6& opbj, const T2* A, const T4& /*beta*/, size_t j, size_t /* i */, size_t /*cind*/, size_t aind, OPA&& opa){return opbj*opa(A[aind+j]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
            else
            {
                using T6 = T3;
                inner_loop(TDM, TDN, _m, _n, _k, _alpha, _A, _lda, _B, _incb, _beta, _C, _ldc, std::forward<OPA>(_opa), std::forward<OPB>(_opb), 
                    [](const T3* B, size_t j, const T1& /*alpha*/, OPB&& opb){return opb(B[j]);},
                    [](const T5* /*C*/, const T6& opbj, const T2* A, const T4& /*beta*/, size_t j, size_t /* i */, size_t /*cind*/, size_t aind, OPA&& opa){return opbj*opa(A[aind+j]);},
                    [](const T5* /*C*/,  const T4& /*beta*/, size_t /*i*/, size_t /*cind*/){return T5(0.0);});
            }
        }
    }
};  //dm_dgm_mt

////////////////////////////////////////////////////////////////////////////////
//Function for computing the trace of a matrix
////////////////////////////////////////////////////////////////////////////////
template <typename T>
T trace(const size_t m, const T* A, const size_t inca)
{
    T ret(0);
    if(inca != 1){for(size_t i=0; i<m; ++i){ret += A[i*inca];}}
    else{for(size_t i=0; i<m; ++i){ret += A[i];}}
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
//Function for computing an elementwise product of two vectors.  This is used //
//in the dgmv function call.                                                  //
////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename OPA, typename OPB>
void elementwise_multiplication(const size_t m, T1 alpha, const T2* A, const size_t inca, const T3* B, const size_t incb, T4 beta, T5* C, const size_t incc, OPA&& opa, OPB&& opb)
{
    if(beta != T4(0.0))
    {
        if(alpha != T1(1.0)){for(size_t i=0; i<m; ++i){size_t ic = i*incc;   C[ic] = alpha*opa(A[i*inca])*opb(B[i*incb]) + beta*C[ic];}}
        else{for(size_t i=0; i<m; ++i){size_t ic = i*incc;   C[ic] = opa(A[i*inca])*opb(B[i*incb]) + beta*C[ic];}}
    }
    else
    {
        if(alpha != T1(1.0)){for(size_t i=0; i<m; ++i){C[i*incc] = alpha*opa(A[i*inca])*opb(B[i*incb]);}}
        else{for(size_t i=0; i<m; ++i){C[i*incc] = opa(A[i*inca])*opb(B[i*incb]);}}
    }
}

namespace csrmm_helper
{
template <bool trans_res, size_t MAXDIM> struct block_assign;
template <size_t MAXDIM> struct block_assign<false, MAXDIM>
{
    using size_type = std::size_t;
    using index_type = int;
    template <typename T, typename ELEM_ASSIGN>
    inline void operator()(const size_type TDM, const size_type TDN, T* C, const size_type bi, const size_type bj, const size_type Ci, const size_type Cj, const size_type ldc, const T alpha, const std::array<T, MAXDIM>& r, const T beta, ELEM_ASSIGN&& elfunc)
    {
        for(size_type ii=0; ii < TDM && ii+bi < Ci; ++ii)
        {
            size_type Cskip = (ii + bi)*ldc + bj;    
            size_type rskip = (ii*TDN);
            for(size_type jj=0; jj < TDN && jj+bj < Cj; ++jj)
            {
                elfunc(C[Cskip+jj], alpha, r[rskip+jj], beta);
            } 
        }
    } 
};
template <size_t MAXDIM> struct block_assign<true, MAXDIM>
{
    using size_type = std::size_t;
    using index_type = int;
    template <typename T, typename ELEM_ASSIGN>
    inline void operator()(const size_type TDM, const size_type TDN, T* C, const size_type bi, const size_type bj, const size_type Ci, const size_type Cj, const size_type ldc, const T alpha, const std::array<T, MAXDIM>& r, const T beta, ELEM_ASSIGN&& elfunc)
    {
        for(size_type jj=0; jj < TDN && jj+bj < Cj; ++jj)
        {
            size_type Cskip = (jj + bj)*ldc+bi;  
            for(size_type ii=0; ii < TDM && ii+bi < Ci; ++ii)
            {
                elfunc(C[Cskip+ii], alpha, r[ii*TDN+jj], beta);
            } 
        }
    } 
};
}   //namespace csrmm_helper


template <size_t MAXM, size_t MAXN>
class csr_dm_m
{
public:
    static constexpr size_t MAXDIM = MAXM*MAXN;
    using size_type = std::size_t;
    using index_type = int;

    //this is almost certainly not the most efficient way to do this at present.  In the future it might be worth combining the load from A to work space, but that is quite difficult to do and so I'm not going to bother with that at 
    //all right now.
    template <bool trans_res, typename T, typename OPA, typename OPB>
    static inline void eval(const size_type _TDM, const size_type _TDN, const size_type m, const size_type n, T alpha, const T* A, const index_type* rowptr, const index_type* colind, const T* B, const size_type ldb, const T beta, T* C, const size_type ldc, OPA&& opa, OPB&& opb)
    {   
        size_type Ci = trans_res ? n : m;
        size_type Cj = trans_res ? m : n;
        size_type TDM = trans_res ? _TDN : _TDM;
        size_type TDN = trans_res ? _TDM : _TDN;

        csrmm_helper::block_assign<trans_res, MAXDIM> bl_ass;
        if(alpha == T(1.0))
        {
            if(beta == T(0.0)){inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& /* _alpha */, const T& _r, const T& /* _beta */) {_C = _r;});} 
            else{inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& /* _alpha */, const T& _r, const T& _beta) {_C = _r + _beta*_C;});}
        }
        else
        {
            if(beta == T(0.0)){inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& _alpha, const T& _r, const T& /* _beta */) {_C = _alpha*_r;});}
            else{inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& _alpha, const T& _r, const T& _beta) {_C = _alpha*_r + _beta*_C;});}
        }
    }

protected:
    template <typename T, typename OPA, typename OPB, typename BLOCK_ASSIGN, typename ELEM_ASSIGN>
    static inline void inner_loop(const size_type TDM, const size_type TDN, const size_type Ci, const size_type Cj, T alpha, const T* A, const index_type* rowptr, const index_type* colind, const T* B, const size_type ldb, const T beta, T* C, const size_type ldc, OPA&& opa, OPB&& opb, BLOCK_ASSIGN&& blfunc, ELEM_ASSIGN&& elfunc)
    {   
        std::array<T,MAXDIM> rbuf; 
#ifdef USE_OPENMP
        #pragma omp parallel for default(shared) private(rbuf)  schedule(static)
#endif
        for(size_type bi=0; bi < Ci; bi += TDM)
        {
            for(size_type bj=0; bj<Cj; bj += TDN)
            {
                //now we fill the rbuf array with zeros
                for(size_type e = 0; e < TDM*TDN; ++e){rbuf[e] = T(0.0);}

                //this is the row of the temporary buffer we are dealing with 
                for(size_type ii=0; ii < TDM && ii+bi < Ci; ++ii)
                {   
                    size_type i = ii + bi;
                    //now run over all of the elements of the csr matrix
                    for(index_type index=rowptr[i]; index<rowptr[i+1]; ++index)
                    {
                        size_type bskip = colind[index]*ldb;
                        //loading the entire row in the TDN of the matrix B into bbuf and str
                        for(size_type jj=0; jj < TDN && jj+bj < Cj; ++jj)
                        {
                            size_type j = jj+bj;
                            rbuf[ii*TDN + jj] += opa(A[index])*opb(B[bskip+j]);
                        }
                    }
                }
                blfunc(TDM, TDN, C, bi, bj, Ci, Cj, ldc, alpha, rbuf, beta, std::forward<ELEM_ASSIGN>(elfunc));
            }
        }
    }
};

template <size_t MAXM, size_t MAXN>
class csr_dm_mt
{
public:
    static constexpr size_t MAXDIM = MAXM*MAXN;
    using size_type = std::size_t;
    using index_type = int;

    template <bool trans_res, typename T, typename OPA, typename OPB>
    static inline void eval(const size_type _TDM, const size_type _TDN, const size_type m, const size_type n, T alpha, const T* A, const index_type* rowptr, const index_type* colind, const T* B, const size_type ldb, const T beta, T* C, const size_type ldc, OPA&& opa, OPB&& opb)
    {   
        size_type Ci = trans_res ? n : m;
        size_type Cj = trans_res ? m : n;
        size_type TDM = trans_res ? _TDN : _TDM;
        size_type TDN = trans_res ? _TDM : _TDN;
        
        csrmm_helper::block_assign<trans_res, MAXDIM> bl_ass;
        if(alpha == T(1.0))
        {
            if(beta == T(0.0)){inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& /* _alpha */, const T& _r, const T& /* _beta */) {_C = _r;});} 
            else{inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& /* _alpha */, const T& _r, const T& _beta) {_C = _r + _beta*_C;});}
        }
        else
        {
            if(beta == T(0.0)){inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& _alpha, const T& _r, const T& /* _beta */) {_C = _alpha*_r;});}
            else{inner_loop(TDM, TDN, Ci, Cj, alpha, A, rowptr, colind, B, ldb, beta, C, ldc, std::forward<OPA>(opa), std::forward<OPB>(opb), bl_ass, [](T& _C, const T& _alpha, const T& _r, const T& _beta) {_C = _alpha*_r + _beta*_C;});}
        }
    }

protected:
    template <typename T, typename OPA, typename OPB, typename BLOCK_ASSIGN, typename ELEM_ASSIGN>
    static inline void inner_loop(const size_type TDM, const size_type TDN, const size_type Ci, const size_type Cj, T alpha, const T* A, const index_type* rowptr, const index_type* colind, const T* B, const size_type ldb, const T beta, T* C, const size_type ldc, OPA&& opa, OPB&& opb, BLOCK_ASSIGN&& blfunc, ELEM_ASSIGN&& elfunc)
    {   
        std::array<T,MAXDIM> rbuf; 
#ifdef USE_OPENMP
        #pragma omp parallel for default(shared) private(rbuf)
#endif
        for(size_type bi=0; bi < Ci; bi += TDM)
        {
            for(size_type bj=0; bj<Cj; bj += TDN)
            {
                //this is the row of the temporary buffer we are dealing with 
                for(size_type ii=0; ii < TDM && ii+bi < Ci; ++ii)
                {   
                    size_type i = ii + bi;
                    for(size_type jj=0; jj < TDN && jj+bj < Cj; ++jj)
                    {
                        size_type bskip = (jj+bj)*ldb;
                        //now run over all of the elements of the csr matrix
                        T rbt = 0.0;
                        for(index_type index=rowptr[i]; index<rowptr[i+1]; ++index){rbt += opa(A[index])*opb(B[bskip+colind[index]]);}
                        rbuf[ii*TDN+jj] = rbt;
                    }
                }
                blfunc(TDM, TDN, C, bi, bj, Ci, Cj, ldc, alpha, rbuf, beta, std::forward<ELEM_ASSIGN>(elfunc));
            }
        }
    }
};

class csrmv
{
public:
    using size_type = std::size_t;
    using index_type = int;

    template <typename T, typename OPA, typename OPB>
    static inline void eval(const size_type m, T alpha, const T* A, const index_type* rowptr, const index_type* colind, const T* B, const size_type incb, const T beta, T* C, const size_type incc, OPA&& opa, OPB&& opb)
    {   
        if(alpha == T(1.0))
        {
            if(beta == T(0.0)){inner_loop(m, alpha, A, rowptr, colind, B, incb, beta, C, incc, std::forward<OPA>(opa), std::forward<OPB>(opb), [](T& _C, const T& /* _alpha */, const T& _r, const T& /* _beta */) {_C = _r;});} 
            else{inner_loop(m, alpha, A, rowptr, colind, B, incb, beta, C, incc, std::forward<OPA>(opa), std::forward<OPB>(opb), [](T& _C, const T& /* _alpha */, const T& _r, const T& _beta) {_C = _r + _beta*_C;});}
        }
        else
        {
            if(beta == T(0.0)){inner_loop(m, alpha, A, rowptr, colind, B, incb, beta, C, incc, std::forward<OPA>(opa), std::forward<OPB>(opb), [](T& _C, const T& _alpha, const T& _r, const T& /* _beta */) {_C = _alpha*_r;});}
            else{inner_loop(m, alpha, A, rowptr, colind, B, incb, beta, C, incc, std::forward<OPA>(opa), std::forward<OPB>(opb), [](T& _C, const T& _alpha, const T& _r, const T& _beta) {_C = _alpha*_r + _beta*_C;});}
        }
    }

protected:
    template <typename T, typename OPA, typename OPB, typename ELEM_ASSIGN>
    static inline void inner_loop(const size_type m, T alpha, const T* A, const index_type* rowptr, const index_type* colind, const T* B, const size_type incb, const T beta, T* C, const size_type incc, OPA&& opa, OPB&& opb, ELEM_ASSIGN&& elfunc)
    {   
#ifdef USE_OPENMP
        #pragma omp parallel for default(shared) schedule(static)
#endif
        for(size_type i=0; i < m; ++i)
        {
            T rbt = 0.0;
            for(index_type index=rowptr[i]; index<rowptr[i+1]; ++index){rbt += opa(A[index])*opb(B[colind[index]*incb]);}
            elfunc(C[i*incc], alpha, rbt, beta);
        }
    }
};


template <typename T> 
T dot(size_t N, const T* const X, size_t INCX,  const T* const Y, size_t INCY)
{
    T ret = T(0.0); ret*=0.0;
    for(size_t i = 0;  i < N; ++i)
    {
        ret += X[i*INCX]*Y[i*INCY];
    }
    return ret;
}

template <typename T> 
T conj_dot(size_t N, const T* const X, size_t INCX,  const T* const Y, size_t INCY)
{
    T ret = T(0.0); ret*=0.0;
    for(size_t i = 0;  i < N; ++i)
    {
        ret += conj(X[i*INCX])*Y[i*INCY];
    }
    return ret;
}

}   //namespace eblas_kernels
}   //namespace linalg

#endif //LINALG_BACKEND_EXTENDED_BLAS_KERNELS_HPP

