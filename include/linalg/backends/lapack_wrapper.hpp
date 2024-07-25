#ifndef LINALG_BACKENDS_LAPACK_WRAPPER_HPP
#define LINALG_BACKENDS_LAPACK_WRAPPER_HPP

namespace linalg
{
namespace lapack
{
#ifndef BLAS_64_BIT
    using blas_int_type = int;
#else
    using blas_int_type = int64_t;
#endif
#ifndef BLAS_HEADER_INCLUDED
extern "C"
{
    //////////////////////////////////////////////////////////////////////////////////////////
    //                    SINGULAR VALUES DECOMPOSITION LAPACK ROUTINES                     //
    //////////////////////////////////////////////////////////////////////////////////////////
    //interface for the standard singular values decomposition implementation in lapack
    void sgesvd_(const char* JOBU, const char* JOBVT, const blas_int_type* M, const blas_int_type* N,           float* A, const blas_int_type* LDA,  float* S,           float* U, const blas_int_type* LDU,           float* VT, const blas_int_type* LDVT,           float* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void dgesvd_(const char* JOBU, const char* JOBVT, const blas_int_type* M, const blas_int_type* N,          double* A, const blas_int_type* LDA, double* S,          double* U, const blas_int_type* LDU,          double* VT, const blas_int_type* LDVT,          double* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void cgesvd_(const char* JOBU, const char* JOBVT, const blas_int_type* M, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  float* S,  complex<float>* U, const blas_int_type* LDU,  complex<float>* VT, const blas_int_type* LDVT,  complex<float>* WORK, const blas_int_type* LWORK,  float* RWORK, blas_int_type* INFO);
    void zgesvd_(const char* JOBU, const char* JOBVT, const blas_int_type* M, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, double* S, complex<double>* U, const blas_int_type* LDU, complex<double>* VT, const blas_int_type* LDVT, complex<double>* WORK, const blas_int_type* LWORK, double* RWORK, blas_int_type* INFO);

    //interface for the divide and conquer singular values decomposition implementation in lapack
    void sgesdd_(const char* JOBZ, const blas_int_type* M, const blas_int_type* N,           float* A, const blas_int_type* LDA,  float* S,           float* U, const blas_int_type* LDU,           float* VT, const blas_int_type* LDVT,           float* WORK, const blas_int_type* LWORK,                blas_int_type* IWORK, blas_int_type* INFO);
    void dgesdd_(const char* JOBZ, const blas_int_type* M, const blas_int_type* N,          double* A, const blas_int_type* LDA, double* S,          double* U, const blas_int_type* LDU,          double* VT, const blas_int_type* LDVT,          double* WORK, const blas_int_type* LWORK,                blas_int_type* IWORK, blas_int_type* INFO);
    void cgesdd_(const char* JOBZ, const blas_int_type* M, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  float* S,  complex<float>* U, const blas_int_type* LDU,  complex<float>* VT, const blas_int_type* LDVT,  complex<float>* WORK, const blas_int_type* LWORK,  float* RWORK, blas_int_type* IWORK, blas_int_type* INFO);
    void zgesdd_(const char* JOBZ, const blas_int_type* M, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, double* S, complex<double>* U, const blas_int_type* LDU, complex<double>* VT, const blas_int_type* LDVT, complex<double>* WORK, const blas_int_type* LWORK, double* RWORK, blas_int_type* IWORK, blas_int_type* INFO);

    //////////////////////////////////////////////////////////////////////////////////////////
    //         QR DECOMPOSITION WITH AND WITHOUT COLUMN PIVOTING LAPACK ROUTINES            //
    //////////////////////////////////////////////////////////////////////////////////////////
    //functions for computing the qr decomposition.  These functions construct the R matrix and the elementary reflectors required to construct the Q matrix.
    void sgeqp3_(const blas_int_type* M, const blas_int_type* N,           float* A, const blas_int_type* LDA, blas_int_type* JPVT,           float* TAU,           float* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void dgeqp3_(const blas_int_type* M, const blas_int_type* N,          double* A, const blas_int_type* LDA, blas_int_type* JPVT,          double* TAU,          double* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void cgeqp3_(const blas_int_type* M, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA, blas_int_type* JPVT,  complex<float>* TAU,  complex<float>* WORK, const blas_int_type* LWORK,  float* RWORK, blas_int_type* INFO);
    void zgeqp3_(const blas_int_type* M, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, blas_int_type* JPVT, complex<double>* TAU, complex<double>* WORK, const blas_int_type* LWORK, double* RWORK, blas_int_type* INFO);

    //functions for computing the qr decomposition.  These functions construct the R matrix and the elementary reflectors required to construct the Q matrix.
    void sgeqrf_(const blas_int_type* M, const blas_int_type* N,           float* A, const blas_int_type* LDA,           float* TAU,           float* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void dgeqrf_(const blas_int_type* M, const blas_int_type* N,          double* A, const blas_int_type* LDA,          double* TAU,          double* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void cgeqrf_(const blas_int_type* M, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  complex<float>* TAU,  complex<float>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void zgeqrf_(const blas_int_type* M, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);

    //these functions take the elementary reflectors generated by geqp3 and generate the corresponding Q matrix
    void sorgqr_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K,           float* A, const blas_int_type* LDA,           float* TAU,           float* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void dorgqr_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K,          double* A, const blas_int_type* LDA,          double* TAU,          double* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void cungqr_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K,  complex<float>* A, const blas_int_type* LDA,  complex<float>* TAU,  complex<float>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void zungqr_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K, complex<double>* A, const blas_int_type* LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);

    //////////////////////////////////////////////////////////////////////////////////////////
    //                           LQ DECOMPOSITION LAPACK ROUTINES                           //
    //////////////////////////////////////////////////////////////////////////////////////////
    //functions for computing the lq decomposition.  These functions construct the R matrix and the elementary reflectors required to construct the Q matrix.
    void sgelqf_(const blas_int_type* M, const blas_int_type* N,           float* A, const blas_int_type* LDA,           float* TAU,           float* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void dgelqf_(const blas_int_type* M, const blas_int_type* N,          double* A, const blas_int_type* LDA,          double* TAU,          double* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void cgelqf_(const blas_int_type* M, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  complex<float>* TAU,  complex<float>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void zgelqf_(const blas_int_type* M, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);

    //these functions take the elementary reflectors generated by gelqf and generate the corresponding Q matrix
    void sorglq_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K,           float* A, const blas_int_type* LDA,           float* TAU,           float* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void dorglq_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K,          double* A, const blas_int_type* LDA,          double* TAU,          double* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void cunglq_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K,  complex<float>* A, const blas_int_type* LDA,  complex<float>* TAU,  complex<float>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void zunglq_(const blas_int_type* M, const blas_int_type* N, const blas_int_type* K, complex<double>* A, const blas_int_type* LDA, complex<double>* TAU, complex<double>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);

    //////////////////////////////////////////////////////////////////////////////////////////
    //                             EIGENSOLVER LAPACK ROUTINES                              //
    //////////////////////////////////////////////////////////////////////////////////////////
    //functions for taking an upper hessenberg matrix and computing its eigenvalues
    void shseqr_(const char* JOB, const char* COMPZ, const blas_int_type* N, const blas_int_type* ILO, const blas_int_type* IHI,           float* H, const blas_int_type* LDH,          float* WR,  float* WI,           float* Z, const blas_int_type* LDZ,           float* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void dhseqr_(const char* JOB, const char* COMPZ, const blas_int_type* N, const blas_int_type* ILO, const blas_int_type* IHI,          double* H, const blas_int_type* LDH,         double* WR, double* WI,          double* Z, const blas_int_type* LDZ,          double* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void chseqr_(const char* JOB, const char* COMPZ, const blas_int_type* N, const blas_int_type* ILO, const blas_int_type* IHI,  complex<float>* H, const blas_int_type* LDH,  complex<float>* W,              complex<float>* Z, const blas_int_type* LDZ,  complex<float>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void zhseqr_(const char* JOB, const char* COMPZ, const blas_int_type* N, const blas_int_type* ILO, const blas_int_type* IHI, complex<double>* H, const blas_int_type* LDH, complex<double>* W,             complex<double>* Z, const blas_int_type* LDZ, complex<double>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);

    //functions for computing the eigenvectors of an upper triangular matrix produced from a schur decomposition 
    void strevc_(const char* SIDE, const char* HOWMNY, const bool* SELECT, const blas_int_type* N,           float* T, const blas_int_type* LDT,           float* VL, const blas_int_type* LDVL,           float* VR, const blas_int_type* LDVR, const blas_int_type* MM, const blas_int_type* M,           float* WORK,                blas_int_type* INFO);
    void dtrevc_(const char* SIDE, const char* HOWMNY, const bool* SELECT, const blas_int_type* N,          double* T, const blas_int_type* LDT,          double* VL, const blas_int_type* LDVL,          double* VR, const blas_int_type* LDVR, const blas_int_type* MM, const blas_int_type* M,          double* WORK,                blas_int_type* INFO);
    void ctrevc_(const char* SIDE, const char* HOWMNY, const bool* SELECT, const blas_int_type* N,  complex<float>* T, const blas_int_type* LDT,  complex<float>* VL, const blas_int_type* LDVL,  complex<float>* VR, const blas_int_type* LDVR, const blas_int_type* MM, const blas_int_type* M,  complex<float>* WORK,  float* RWORK, blas_int_type* INFO);
    void ztrevc_(const char* SIDE, const char* HOWMNY, const bool* SELECT, const blas_int_type* N, complex<double>* T, const blas_int_type* LDT, complex<double>* VL, const blas_int_type* LDVL, complex<double>* VR, const blas_int_type* LDVR, const blas_int_type* MM, const blas_int_type* M, complex<double>* WORK, double* RWORK, blas_int_type* INFO);

    //functions for computing the eigenvalues and eigenvectors of a general matrix
    void sgeev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N,           float* A, const blas_int_type* LDA,          float* WR,  float* WI,           float* VL, const blas_int_type* LDVL,           float* VR, const blas_int_type* LDVR,           float* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void dgeev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N,          double* A, const blas_int_type* LDA,         double* WR, double* WI,          double* VL, const blas_int_type* LDVL,          double* VR, const blas_int_type* LDVR,          double* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void cgeev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  complex<float>* W,              complex<float>* VL, const blas_int_type* LDVL,  complex<float>* VR, const blas_int_type* LDVR,  complex<float>* WORK, const blas_int_type* LWORK,  float* RWORK, blas_int_type* INFO);
    void zgeev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, complex<double>* W,             complex<double>* VL, const blas_int_type* LDVL, complex<double>* VR, const blas_int_type* LDVR, complex<double>* WORK, const blas_int_type* LWORK, double* RWORK, blas_int_type* INFO);

    //functions for computing the eigenvalues of hermitian matrices (real symmetric in the case of real matrices)
    void ssyev_(const char* JOBZ, const char* UPLO, const blas_int_type* N,            float* A, const blas_int_type* LDA,  float* W,            float* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void dsyev_(const char* JOBZ, const char* UPLO, const blas_int_type* N,           double* A, const blas_int_type* LDA, double* W,           double* WORK, const blas_int_type* LWORK,                blas_int_type* INFO);
    void cheev_(const char* JOBZ, const char* UPLO, const blas_int_type* N,   complex<float>* A, const blas_int_type* LDA,  float* W,   complex<float>* WORK, const blas_int_type* LWORK,  float* RWORK, blas_int_type* INFO);
    void zheev_(const char* JOBZ, const char* UPLO, const blas_int_type* N,  complex<double>* A, const blas_int_type* LDA, double* W,  complex<double>* WORK, const blas_int_type* LWORK, double* RWORK, blas_int_type* INFO);

    //functions for diagonalising a symmetric tridiagonal matrix
    void sstev_(const char* COMPZ, const blas_int_type* N,  float* D,  float* E,  float* Z, const blas_int_type* LDZ,  float* WORK, blas_int_type* INFO);
    void dstev_(const char* COMPZ, const blas_int_type* N, double* D, double* E, double* Z, const blas_int_type* LDZ, double* WORK, blas_int_type* INFO);
    //void cstedc_(const char* COMPZ, const blas_int_type* N,  complex<float>* D,  complex<float>* E,  complex<float>* Z, const blas_int_type* LDZ,  complex<float>* WORK, const blas_int_type* LWORK,  float* RWORK, const blas_int_type* LRWORK, blas_int_type* IWORK, const blas_int_type* LIWORK, blas_int_type* INFO);
    //void zstedc_(const char* COMPZ, const blas_int_type* N,  complex<double>* D,  complex<double>* E,  complex<double>* Z, const blas_int_type* LDZ,  complex<double>* WORK, const blas_int_type* LWORK,  double* RWORK, const blas_int_type* LRWORK, blas_int_type* IWORK, const blas_int_type* LIWORK, blas_int_type* INFO);
    

    //function for computing a symmetric tridiagonalisation of a general hermitian matrix
    void ssytrd_(const char* UPLO, const blas_int_type* N,           float* A, const blas_int_type* LDA,  float* D,  float* E,           float* TAU,           float* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void dsytrd_(const char* UPLO, const blas_int_type* N,          double* A, const blas_int_type* LDA, double* D, double* E,          double* TAU,          double* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void chetrd_(const char* UPLO, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  float* D,  float* E,  complex<float>* TAU,  complex<float>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);
    void zhetrd_(const char* UPLO, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, double* D, double* E, complex<double>* TAU, complex<double>* WORK, const blas_int_type* LWORK, blas_int_type* INFO);


    //function for computing the action of the orthogonal/unitary matrix Q that is defined as the product of elementary reflectors that
    //is returned from the ?sytrd and ?hetrd routines
    void sormtr_(const char* side, const char* UPLO, const char* TRANS, const blas_int_type* M,  const blas_int_type* N,           float* A, const blas_int_type* LDA,           float* TAU,           float* C, const blas_int_type* LDC,           float* work, const blas_int_type* LWORK, blas_int_type* INFO);
    void dormtr_(const char* side, const char* UPLO, const char* TRANS, const blas_int_type* M,  const blas_int_type* N,          double* A, const blas_int_type* LDA,          double* TAU,          double* C, const blas_int_type* LDC,          double* work, const blas_int_type* LWORK, blas_int_type* INFO);
    void cunmtr_(const char* side, const char* UPLO, const char* TRANS, const blas_int_type* M,  const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  complex<float>* TAU,  complex<float>* C, const blas_int_type* LDC,  complex<float>* work, const blas_int_type* LWORK, blas_int_type* INFO);
    void zunmtr_(const char* side, const char* UPLO, const char* TRANS, const blas_int_type* M,  const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, complex<double>* TAU, complex<double>* C, const blas_int_type* LDC, complex<double>* work, const blas_int_type* LWORK, blas_int_type* INFO);

    //functions for computing the LU decomposition of a matrix
    void sgetrf_(const blas_int_type* M, const blas_int_type* N,           float* A, const blas_int_type* LDA, blas_int_type* IPIV, blas_int_type* INFO);
    void dgetrf_(const blas_int_type* M, const blas_int_type* N,          double* A, const blas_int_type* LDA, blas_int_type* IPIV, blas_int_type* INFO);
    void cgetrf_(const blas_int_type* M, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA, blas_int_type* IPIV, blas_int_type* INFO);
    void zgetrf_(const blas_int_type* M, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, blas_int_type* IPIV, blas_int_type* INFO);

    //functions for solving a linear system of equations
    void sgetrs_(const char* TRANS, const blas_int_type* N, const blas_int_type* NRHS,           float* A, const blas_int_type* LDA, blas_int_type* IPIV,           float* B, const blas_int_type* LDB, blas_int_type* INFO);
    void dgetrs_(const char* TRANS, const blas_int_type* N, const blas_int_type* NRHS,          double* A, const blas_int_type* LDA, blas_int_type* IPIV,          double* B, const blas_int_type* LDB, blas_int_type* INFO);
    void cgetrs_(const char* TRANS, const blas_int_type* N, const blas_int_type* NRHS,  complex<float>* A, const blas_int_type* LDA, blas_int_type* IPIV,  complex<float>* B, const blas_int_type* LDB, blas_int_type* INFO);
    void zgetrs_(const char* TRANS, const blas_int_type* N, const blas_int_type* NRHS, complex<double>* A, const blas_int_type* LDA, blas_int_type* IPIV, complex<double>* B, const blas_int_type* LDB, blas_int_type* INFO);

    //functions for computing the generalised eigenvalues and eigenvectors of a general matrix
    void sggev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N,           float* A, const blas_int_type* LDA,           float* B, const blas_int_type* LDB,          float* ALPHAR,  float* ALPHAI,           float* BETA,           float* VL, const blas_int_type* LDVL,           float* VR, const blas_int_type* LDVR,           float* WORK, const blas_int_type* LWORK,                      blas_int_type* INFO);
    void dggev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N,          double* A, const blas_int_type* LDA,          double* B, const blas_int_type* LDB,         double* ALPHAR, double* ALPHAI,          double* BETA,          double* VL, const blas_int_type* LDVL,          double* VR, const blas_int_type* LDVR,          double* WORK, const blas_int_type* LWORK,                      blas_int_type* INFO);
    void cggev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N,  complex<float>* A, const blas_int_type* LDA,  complex<float>* B, const blas_int_type* LDB,  complex<float>* ALPHA,                  complex<float>* BETA,  complex<float>* VL, const blas_int_type* LDVL,  complex<float>* VR, const blas_int_type* LDVR,  complex<float>* WORK, const blas_int_type* LWORK,  const float* RWORK, blas_int_type* INFO);
    void zggev_(const char* JOBVL, const char* JOBVR, const blas_int_type* N, complex<double>* A, const blas_int_type* LDA, complex<double>* B, const blas_int_type* LDB, complex<double>* ALPHA,                 complex<double>* BETA, complex<double>* VL, const blas_int_type* LDVL, complex<double>* VR, const blas_int_type* LDVR, complex<double>* WORK, const blas_int_type* LWORK, const double* RWORK, blas_int_type* INFO);

}
#endif


}   //namespace lapack
}   //namespace linalg

#endif  //LINALG_BACKENDS_LAPACK_WRAPPER_HPP//


